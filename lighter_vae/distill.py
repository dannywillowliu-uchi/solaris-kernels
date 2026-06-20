"""Distill a slim VAE decoder to match the frozen Wan-VAE (same latent space => drop-in).

Teacher = get_vae_model() + vae.pt (dim 96, z_dim 16, dim_mult [1,2,4,4], 2 res-blocks).
Student = WanVAE_ with a SLIM config (dim 48, dim_mult [1,2,2,2], 1 res-block, SAME z_dim 16).
Only z_dim/VAE_SCALE matter for the interface, so the student is a drop-in faster decoder.

Run inside the Solaris repo env on a GPU box:
  SOLARIS_RUN=/path GPU=0 venv/bin/python distill.py --steps 2000 --out student_dec.ckpt

First pass uses RANDOM latents (proves the pipeline + a first decoder). For shipping quality,
swap `sample_latents` to encode real eval frames (teacher.encode) so the student matches the
serving latent distribution. Gate on SSIM/PSNR vs teacher + decode-fps before swapping in.
"""

from __future__ import annotations

import argparse
import os
import time

import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
from flax import nnx

from src.models.model_loaders import get_vae_model
from src.models.wan_vae import WanVAE_, VAE_SCALE

B = os.environ.get("SOLARIS_RUN", "/mnt/SFS-nc15dnf9/oasis-port/solaris-run")
VAE_CKPT = f"{B}/solaris/pretrained/vae.pt"

# latent geometry (b t h w c). H,W = encoder-downsampled spatial; C = z_dim.
LAT = dict(b=1, t=4, h=44, w=80, c=16)
STUDENT_CFG = dict(dim=48, z_dim=16, dim_mult=[1, 2, 2, 2], num_res_blocks=1,
	attn_scales=[], temperal_downsample=[False, True, True], dropout=0.0)


def load_teacher():
	t = get_vae_model()
	g, s = nnx.split(t)
	s = ocp.StandardCheckpointer().restore(VAE_CKPT, s)
	return nnx.merge(g, s)


def sample_latents(key):
	# random latents at the natural scale; TODO: replace with teacher.encode(real eval frames)
	return jax.random.normal(key, (LAT["b"], LAT["t"], LAT["h"], LAT["w"], LAT["c"])) * 0.5


def main() -> None:
	ap = argparse.ArgumentParser()
	ap.add_argument("--steps", type=int, default=2000)
	ap.add_argument("--lr", type=float, default=1e-4)
	ap.add_argument("--out", default=f"{B}/student_dec.ckpt")
	args = ap.parse_args()

	teacher = load_teacher()
	student = WanVAE_(rngs=nnx.Rngs(0), **STUDENT_CFG)
	# head-start: copy the 1x1x1 conv2 (z_dim->z_dim, identical shape) from teacher
	try:
		ts = nnx.state(teacher); ss = nnx.state(student)
		ss["conv2"] = ts["conv2"]; nnx.update(student, ss)
	except Exception as e:
		print("conv2 copy skipped:", e)

	opt = nnx.Optimizer(student, optax.adam(args.lr))

	@nnx.jit
	def train_step(student, opt, z, target):
		def loss_fn(m):
			pred = m.decode(z, VAE_SCALE)
			return jnp.mean(jnp.abs(pred.astype(jnp.float32) - target.astype(jnp.float32)))
		loss, grads = nnx.value_and_grad(loss_fn)(student)
		opt.update(grads)
		return loss

	key = jax.random.PRNGKey(0)
	t0 = time.time()
	for step in range(args.steps):
		key, k = jax.random.split(key)
		z = sample_latents(k)
		target = jax.lax.stop_gradient(teacher.decode(z, VAE_SCALE))
		loss = train_step(student, opt, z, target)
		if step % 100 == 0:
			loss.block_until_ready()
			print(f"step {step} L1={float(loss):.4f} ({(time.time()-t0):.0f}s)", flush=True)

	# validate: quality (1 - relL2) + decode speed vs teacher
	z = sample_latents(jax.random.PRNGKey(123))
	tgt = teacher.decode(z, VAE_SCALE); prd = student.decode(z, VAE_SCALE)
	rel = float(jnp.linalg.norm((prd - tgt).astype(jnp.float32)) / (jnp.linalg.norm(tgt.astype(jnp.float32)) + 1e-6))
	def _t(fn):
		fn().block_until_ready()
		s = time.time()
		for _ in range(20):
			r = fn()
		r.block_until_ready(); return (time.time() - s) / 20 * 1000
	t_teacher = _t(lambda: teacher.decode(z, VAE_SCALE))
	t_student = _t(lambda: student.decode(z, VAE_SCALE))
	print(f"VALIDATE rel_l2={rel:.4f} (lower=better)  teacher={t_teacher:.2f}ms student={t_student:.2f}ms "
		f"speedup={t_teacher/t_student:.2f}x", flush=True)

	ocp.StandardCheckpointer().save(args.out, nnx.state(student))
	print(f"saved student -> {args.out}", flush=True)


if __name__ == "__main__":
	main()
