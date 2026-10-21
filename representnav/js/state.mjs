export const FUTURE_TIMES = Object.freeze([0.4, 0.8, 1.2, 1.6]);
export const MODALITIES = Object.freeze([
  { id: 'occ', name: '3D occupancy', detail: 'Geometry of the scene', dimension: '120 × 80 × 20 voxels', icon: '⬡' },
  { id: 'semantic', name: 'Semantics', detail: 'A world of meaningful regions', dimension: '64 × 64 semantic readout', icon: '▧' },
  { id: 'box', name: 'Target localization', detail: 'Where the target is', dimension: 'Target box on RGB', icon: '⌗' },
  { id: 'ego', name: 'Ego motion', detail: 'The preceding 0.6 seconds', dimension: 'Past motion · selected robot frame · meters', icon: '↗' },
  { id: 'target', name: 'Target motion', detail: 'Where the target is heading', dimension: 'Fixed current-robot frame · meters', icon: '⇢' },
  { id: 'depth', name: 'Depth', detail: 'Near and far', dimension: '32 × 32 relative inverse depth', icon: '◫' },
]);

export function createState(caseIds) {
  if (!caseIds.length) throw new Error('At least one case is required.');
  return {
    caseId: caseIds.includes('track') ? 'track' : caseIds[0],
    representation: { modality: 'occ', frame: 0 },
    dream: { modality: 'occ', frame: 1 },
  };
}

export function reduceState(state, action, caseIds) {
  if (action.type === 'case') {
    if (!caseIds.includes(action.id)) return state;
    return { ...state, caseId: action.id, representation: { ...state.representation, frame: 0 }, dream: { ...state.dream, frame: 1 } };
  }
  if (action.type === 'token') {
    if (!Number.isInteger(action.frame)) return state;
    return { ...state, dream: { ...state.dream, frame: Math.max(1, Math.min(4, action.frame)) } };
  }
  if (action.type === 'modality' && ['representation', 'dream'].includes(action.section) && MODALITIES.some(m => m.id === action.id)) {
    return { ...state, [action.section]: { ...state[action.section], modality: action.id } };
  }
  return state;
}

export function animationState(section, progress) {
  const p = Math.max(0, Math.min(1, progress));
  if (section === 'representation') {
    const stage = p < 2 / 12 ? 0 : p < 5 / 12 ? 1 : p < 7 / 12 ? 2 : p < 9 / 12 ? 3 : 4;
    return { stage, token: null, status: [
      'Observe the scene from multiple views',
      'Gather semantic, geometric, and dynamic priors',
      'Construct the navigation representation',
      'Focus on one per-view [CLS]',
      'Explore the information held in the representation',
    ][stage] };
  }
  const stage = p < 3 / 18 ? 0 : p < 6 / 18 ? 1 : p < 12 / 18 ? 2 : p < 16 / 18 ? 3 : 4;
  const token = p < 6 / 18 ? 1 : p < 12 / 18 ? Math.min(4, 1 + Math.floor((p * 18 - 6) / 1.5))
    : p < 16 / 18 ? Math.min(4, 1 + Math.floor(p * 18 - 12)) : 4;
  return { stage, token, status: [
    'An observation and an instruction',
    'Bring language and visual representations together',
    `Imagine the future, one continuous state at a time · h${token}`,
    `Decode the selected future state · +${FUTURE_TIMES[token - 1].toFixed(1)} s`,
    'From an unfolding world to navigation waypoints',
  ][stage] };
}

export function validateManifest(manifest) {
  if (!manifest || !Array.isArray(manifest.cases) || !manifest.cases.length) throw new Error('No cases in the scene manifest.');
  const ids = new Set();
  for (const item of manifest.cases) {
    if (!item.id || ids.has(item.id)) throw new Error('Case IDs must be present and unique.');
    ids.add(item.id);
    const hasOccupancy = !item.availableModalities || item.availableModalities.includes('occ');
    if (!item.input || (hasOccupancy && item.grid?.dims?.join(',') !== '120,80,20')) throw new Error(`Invalid input or voxel grid: ${item.id}`);
    if (item.frames?.length !== 5) throw new Error(`Five scene states are required: ${item.id}`);
    item.frames.forEach((frame, index) => {
      const expected = item.timeBasis === 'video-frame' ? index : index * .4;
      if (Math.abs(frame.time - expected) > 1e-6) throw new Error(`Incorrect future time: ${item.id}/${index}`);
      const required = ['scene', ...(hasOccupancy ? ['voxels'] : []),
        ...['semantic', 'depth'].filter(key => !item.availableModalities || item.availableModalities.includes(key))];
      for (const key of required) if (!frame[key]) throw new Error(`Missing ${key}: ${item.id}/${index}`);
    });
  }
  return manifest;
}
