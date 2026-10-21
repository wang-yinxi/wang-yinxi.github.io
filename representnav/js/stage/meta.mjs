// Chapter metadata for the two stages; kept free of three.js so the page UI can load it early.
export const REP_STEPS = [
  { t: 0, label: 'Observe', text: 'Four synchronized camera views form the observation oₜ.' },
  { t: 1.9, label: 'Frozen priors', text: 'Semantic, geometric and dynamic foundation models stay frozen; features from every depth are kept.' },
  { t: 3.9, label: 'Attend over depth', text: 'At each layer, α scores every encoder layer and every layer below with a learned query w, then mixes them by a softmax over depth.' },
  { t: 5.6, label: 'Tokens', text: 'The attended features become patch tokens; a learnable [CLS] joins them.' },
  { t: 6.4, label: 'Self-attention', text: 'Inside each Representation Residuals layer, the [CLS] and the patch tokens attend to one another.' },
  { t: 8.3, label: 'Three layers', text: 'Each of the three layers repeats it: attend over depth, then self-attention. Each layer’s output joins the bus for the layers above.' },
  { t: 15.5, label: 'Focus on [CLS]', text: 'The [CLS] token gathers each view into one vector: the self-constructed representation.' },
  { t: 16.9, label: 'Decode', text: 'Post-hoc readouts open the token: what the representation holds about the scene.' },
];
export const REP_END = 20.1;

export const DREAM_STEPS = [
  { t: 0, label: 'Observe & instruct', text: 'An instruction and the current observation enter the model.' },
  { t: 2.0, label: 'Context', text: 'Text tokens and self-constructed representation tokens fill the context, then (latent_start).' },
  { t: 4.3, label: 'Latent reasoning', text: 'The LLM generates h₁…h₄; each latent is fed back as the next input embedding.' },
  { t: 11.3, label: 'Act', text: '(latent_end) follows the rollout; the planning head maps it to eight waypoints.' },
  { t: 12.4, label: 'Training targets', text: 'Training only: each h_k predicts the future [CLS] tokens (ℒpred), and VISReg keeps the target space from collapsing.' },
  { t: 14.2, label: 'Emerge', text: 'Decoded post hoc (visualization only), each latent h_k unfolds the scene a moment ahead: 3D occupancy, depth and semantics emerge from the dream.' },
  { t: 23.2, label: 'Full picture', text: 'Back to the whole model: the planner acts on the latents; decoding them is only for visualization.' },
];
export const DREAM_END = 25.8;
