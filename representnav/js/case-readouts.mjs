/** Select one branch without changing its RGB, timestamp, or coordinate frame. */
export function decodedFrame(frame, section = 'representation') {
  if (section !== 'dream') return frame;
  const decoded = { ...frame };
  for (const field of ['semantic', 'depth', 'box', 'voxels', 'fallback', 'egoMotion', 'targetMotion']) {
    const key = `latent${field[0].toUpperCase()}${field.slice(1)}`;
    if (Object.hasOwn(frame, key)) decoded[field] = frame[key];
  }
  if (frame.views) decoded.views = frame.views.map(view => decodedFrame(view, section));
  return decoded;
}

const HAS_READOUT = {
  occ: frame => frame.voxels,
  semantic: frame => frame.semantic || frame.views?.some(view => view.semantic),
  box: frame => frame.box || frame.views?.some(view => view.box),
  ego: frame => frame.egoMotion?.length,
  target: frame => frame.targetMotion?.length,
  depth: frame => frame.depth || frame.views?.some(view => view.depth),
};

/** Readouts the case declares and actually has data for (e.g. no target-motion tab without a target trajectory). */
export function availableModalities(item) {
  const declared = item.availableModalities || ['occ', 'semantic', 'box', 'ego', 'target', 'depth'];
  if (!item.frames?.length) return declared;
  const present = declared.filter(id => !HAS_READOUT[id] || item.frames.some(frame => HAS_READOUT[id](frame)));
  return present.length ? present : declared;
}

export function signedTime(time) {
  return `${time < -1e-6 ? '−' : time > 1e-6 ? '+' : ''}${Math.abs(time).toFixed(1)}`;
}

export function stateTime(item, frame) {
  return item.timeBasis === 'video-frame' ? `Frame ${frame.sourceFrameIndex}` : `${signedTime(frame.time)} s`;
}

export function historyWindow(frame, timeBasis = frame.timeBasis) {
  const times = frame.motionTimes || [frame.time - .6, frame.time];
  if (timeBasis === 'video-frame') return `Frames ${times[0]} to ${times.at(-1)}`;
  return `${signedTime(times[0])} to ${signedTime(times.at(-1))} s`;
}
