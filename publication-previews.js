// Keep the method figure visible until the preview video is ready to play.
document.querySelectorAll('.publication-preview').forEach((preview) => {
  const video = preview.querySelector('video');
  let hovered = false;
  let focused = false;
  let active = false;
  let request = 0;

  const load = () => {
    if (video.hasAttribute('src')) return;
    video.muted = true;
    video.preload = 'metadata';
    video.src = video.dataset.src;
    video.load();
  };

  const stop = () => {
    active = false;
    request += 1;
    video.pause();
    preview.classList.remove('is-playing');
    if (video.readyState > 0) video.currentTime = 0;
  };

  const update = () => {
    if ((!hovered && !focused) || document.hidden) {
      stop();
      return;
    }
    if (active) return;
    active = true;
    const currentRequest = ++request;
    load();
    video.play().catch(() => {
      if (request === currentRequest) stop();
    });
  };

  video.addEventListener('playing', () => {
    if (active) preview.classList.add('is-playing');
    else video.pause();
  });
  video.addEventListener('error', stop);
  preview.addEventListener('pointerenter', (event) => {
    if (event.pointerType === 'touch') return;
    hovered = true;
    update();
  });
  preview.addEventListener('pointerleave', () => {
    hovered = false;
    update();
  });
  preview.addEventListener('focus', () => {
    focused = preview.matches(':focus-visible');
    update();
  });
  preview.addEventListener('blur', () => {
    focused = false;
    update();
  });
  document.addEventListener('visibilitychange', update);
  window.addEventListener('pagehide', stop);

  if ('IntersectionObserver' in window) {
    const observer = new IntersectionObserver((entries) => {
      if (entries[0].isIntersecting) {
        load();
      } else {
        hovered = false;
        stop();
      }
    }, { rootMargin: '100px' });
    observer.observe(preview);
  }
});
