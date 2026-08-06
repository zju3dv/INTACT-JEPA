# Method Notes

## 1. Problem

Let an encoder map an observation to a latent state, $z_t=E_\theta(o_t)$.
A conventional latent world model learns a forward conditional

$$
F_\psi(z_t,a_t)\approx z_{t+1},
$$

but obtains actions for a desired endpoint by numerically inverting $F_\psi$
at deployment. INTACT instead learns a deployable conditional action law
jointly with the representation.

## 2. One Input Grammar, Two Intent Instances

For an observed transition and a future goal from the same demonstration,
INTACT constructs

$$
m_t^{\mathrm{local}}=z_{t+1}-z_t,
\qquad
m_t^{\mathrm{goal}}=\mathrm{sg}(z_g)-z_t.
$$

Both are instances of exactly one predictor input grammar:

$$
x_t(m_t)=[z_t,m_t,z_t\odot m_t,A(a_{t-1})],
\qquad
G_\eta(x_t(m_t))\mapsto p_\eta(a_t\mid x_t(m_t)).
$$

The first-order intent and its state-intent interaction form a matched input
grammar within each call. The interaction is a fixed bilinear feature, not a
learned second-order dynamics model. The local and goal cases are not different
input types or separate actors; only the value and gradient role of $m_t$
change.

## 3. Gradient Contract

The two calls are parameter-isomorphic but not gradient-symmetric.

| Path | Endpoint | Gradient behavior | Purpose |
|---|---|---|---|
| Physical intent | observed $z_{t+1}$ | current and successor latents attached | ground reachability and preserve action-recoverable change |
| Goal intent | future $z_g$ | current latent attached, future goal detached | train the condition available before acting without moving the goal anchor |

There is no pointwise $\lVert m_t^{\mathrm{goal}}-m_t^{\mathrm{local}}\rVert_2$
penalty. Two conditions may be action-equivalent without being close in latent
coordinates.

The statistical unit is one supported triplet $(z_t,m_t,a_t)$: one endpoint
condition and one proper NLL. Local and goal triplets may share the same
demonstrated action, but INTACT never treats endpoint type itself as an
equivalence label.

## 4. Objective

The complete objective combines the unchanged forward prediction term,
representation regularization, and two calls to one proper conditional action
likelihood:

$$
\mathcal L =
\mathcal L_{\mathrm{forward}}
+\lambda_{\mathrm{reg}}\mathcal L_{\mathrm{SIGReg}}
+\lambda_{\mathrm{local}}\mathcal L_{\mathrm{NLL}}^{\mathrm{local}}
+\lambda_{\mathrm{goal}}\mathcal L_{\mathrm{NLL}}^{\mathrm{goal}}.
$$

The exact task-specific and shared-encoder configurations will be released with
the code. Published result tables should always identify which configuration,
checkpoint epoch, and inference interface generated each number.

## 5. Conditional Action Quotient

At fixed current state $z$, define

$$
y\sim_z y'
\quad\Longleftrightarrow\quad
p_E^\star(a\mid z,y)=p_E^\star(a\mid z,y').
$$

The quotient groups conditions by the expert action law they induce. Its image
under the conditional action operator is canonically isomorphic to the set of
realizable action-law families. This is a behavioral correspondence, not a
claim that every latent coordinate is unique or globally identifiable.

For finite data and continuous control, the operational version is local: with
a task-appropriate action metric and tolerance $\epsilon$, predictions inside
the same neighborhood $[a_t]_\epsilon$ may be behaviorally equivalent. The
proper likelihood learns probability mass over that neighborhood instead of
requiring an exact pointwise match. This gives direct readout tolerance to
small prediction errors; it does not identify unsupported goals or guarantee
robustness outside the demonstrated action manifold.

## 6. Deployment Interfaces

**Direct.** Encode the current observation and goal, predict one action chunk,
roll it through the forward model, and replan from the next real observation.
No candidate sequence or terminal latent cost is evaluated.

**Guarded local verification.** Center a small raw-action CEM distribution on
the coherent Direct plan. The world model verifies a local neighborhood; it no
longer discovers the plan from a broad random proposal.

**Actor-disabled CEM.** Disable all INTACT action heads and run the original CEM
planner. This isolates whether action supervision improved the encoder-forward
stack before direct readout.

## 7. What INTACT Is Not

- It is not inverse dynamics alone: the deployable goal call is essential.
- It is not goal-only behavior cloning: the attached physical call and forward
  JEPA jointly shape the representation.
- It is not a post-hoc sampling prior: the encoder and action interface train
  end to end in one stage.
- It does not force a goal endpoint to imitate a one-step successor.
- It does not require globally straight or linear latent trajectories.

## 8. Released Implementation Contract

The clean release uses a ViT-Tiny/14 encoder with a 192-dimensional projected
latent, a depth-6 Forward Predictor, and a depth-3 diagonal-Gaussian intent
Actor. Both intent calls have the exact four-slot grammar

```text
[z_t, m_t, z_t * m_t, A(a_{t-1})].
```

Physical and goal supervision both cover indices 0--6 of the effective
eight-frame window. On all seven indices, the same demonstrated action
contributes both a physical and a goal NLL through the shared Actor. The
implementation folds all 7+7 rows into one vectorized Actor call.

`A(a_{t-1})` follows the boundary-aware previous-action contract. An interior
clip uses the real primitive action chunk immediately before its start. At a
true episode boundary, only unavailable history is left-padded with raw zero
commands; those commands are then transformed using statistics fitted only on
real demonstration actions. Evaluation applies the same raw-coordinate reset
rule before z-scoring. The full definition and the incompatibility with older
paper checkpoints are recorded in
[`PREVIOUS_ACTION_BOUNDARY_20260804.md`](PREVIOUS_ACTION_BOUNDARY_20260804.md).

The task-specific baseline objective is

```text
L = 1.0 * L_forward + 0.02 * L_SIGReg
  + 0.10 * L_local_NLL + 0.05 * L_goal_NLL.
```

Shared E5 training changes only the SIGReg coefficient to `0.03` and uses
fused AdamW with constant learning rate `5e-4`, weight decay `1e-3`, batch 256
per task, five epochs, and Math SDPA. The Forward loss covers all seven
adjacent transitions in the eight-frame window.

Direct and Actor-CEM require a real action history. Pure CEM does not call the
Actor; its zero-mean proposal is only a search distribution and is never
inserted into an Actor input or stored as a checkpoint ablation option.
