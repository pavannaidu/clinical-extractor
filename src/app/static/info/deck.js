(() => {
  'use strict';

  const deckShell = document.getElementById('deck-shell');
  const stageViewport = document.getElementById('main-content');
  const slides = Array.from(document.querySelectorAll('.slide'));
  const previousButton = document.getElementById('previous-slide');
  const nextButton = document.getElementById('next-slide');
  const narrativeRail = document.getElementById('narrative-rail');
  const slidePosition = document.getElementById('slide-position');
  const viewToggle = document.getElementById('view-toggle');
  const fullscreenButton = document.getElementById('fullscreen-button');
  const readerIndexList = document.getElementById('reader-index-list');
  const liveStatus = document.getElementById('live-status');

  if (!deckShell || !stageViewport || !slides.length) return;

  const narrowScreen = window.matchMedia('(max-width: 1024px)');
  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
  const baseTitle = document.body.dataset.deckTitle || document.title;
  const models = slides.map((slide, index) => ({
    slide,
    id: slide.id,
    index,
    short: slide.dataset.short || `Slide ${index + 1}`,
    section: slide.dataset.section?.trim() || 'Deck',
    title: document.getElementById(slide.getAttribute('aria-labelledby'))?.textContent.trim() || `Slide ${index + 1}`
  }));

  const narrativeGroups = [];

  let currentIndex = 0;
  let manualView = null;
  let chromeTimer = null;
  let fullscreenTrigger = null;
  let restoreReaderAfterPrint = false;

  const displayNumber = (index) => String(index + 1).padStart(2, '0');
  const readerMode = () => document.body.classList.contains('reader-mode');

  const buildNarrativeRail = () => {
    if (!narrativeRail) return;

    models.forEach((model) => {
      const previousGroup = narrativeGroups.at(-1);
      if (previousGroup?.name === model.section) previousGroup.models.push(model);
      else narrativeGroups.push({ name: model.section, models: [model] });
    });

    const repeatedSections = narrativeGroups
      .map((group) => group.name)
      .filter((name, index, names) => names.indexOf(name) !== index);
    if (repeatedSections.length) {
      console.warn('Narrative rail sections should be contiguous:', [...new Set(repeatedSections)]);
    }

    narrativeGroups.forEach((group) => {
      const groupElement = document.createElement('div');
      groupElement.className = 'narrative-section';
      groupElement.style.setProperty('--section-span', String(group.models.length));

      const sectionButton = document.createElement('button');
      sectionButton.className = 'narrative-section-button';
      sectionButton.type = 'button';
      sectionButton.title = `Go to ${group.name}`;
      sectionButton.setAttribute('aria-label', `Go to section ${group.name}`);

      const sectionLabel = document.createElement('span');
      sectionLabel.className = 'narrative-section-label';
      sectionLabel.textContent = group.name;

      const sectionTrack = document.createElement('span');
      sectionTrack.className = 'narrative-section-track';
      sectionTrack.setAttribute('aria-hidden', 'true');
      const sectionFill = document.createElement('span');
      sectionFill.className = 'narrative-section-fill';
      sectionTrack.append(sectionFill);
      sectionButton.append(sectionLabel, sectionTrack);
      sectionButton.addEventListener('click', () => goTo(group.models[0].index));

      const slideChoices = document.createElement('div');
      slideChoices.className = 'narrative-slide-choices';
      group.models.forEach((model) => {
        const slideButton = document.createElement('button');
        slideButton.className = 'narrative-slide-button';
        slideButton.type = 'button';
        slideButton.title = `${displayNumber(model.index)} · ${model.title}`;
        slideButton.setAttribute('aria-label', `Go to slide ${model.index + 1}: ${model.title}`);

        const dot = document.createElement('span');
        dot.className = 'narrative-slide-dot';
        dot.setAttribute('aria-hidden', 'true');
        const label = document.createElement('span');
        label.className = 'narrative-slide-label';
        label.textContent = model.short;
        slideButton.append(dot, label);
        slideButton.addEventListener('click', () => goTo(model.index));
        slideChoices.append(slideButton);
        model.railButton = slideButton;
      });

      groupElement.append(sectionButton, slideChoices);
      narrativeRail.append(groupElement);
      Object.assign(group, {
        element: groupElement,
        sectionButton,
        sectionFill,
        firstIndex: group.models[0].index,
        lastIndex: group.models.at(-1).index
      });
    });

    narrativeRail.classList.toggle('is-compact', models.length > 8 || narrativeGroups.length > 5);
  };

  const paintNarrativeRail = () => {
    narrativeGroups.forEach((group) => {
      const isCurrent = currentIndex >= group.firstIndex && currentIndex <= group.lastIndex;
      const isPast = currentIndex > group.lastIndex;
      const completion = isPast
        ? 1
        : isCurrent
          ? (currentIndex - group.firstIndex + 1) / group.models.length
          : 0;

      group.element.classList.toggle('is-current', isCurrent);
      group.element.classList.toggle('is-past', isPast);
      group.sectionFill.style.width = `${completion * 100}%`;
      if (isCurrent) group.sectionButton.setAttribute('aria-current', 'step');
      else group.sectionButton.removeAttribute('aria-current');

      group.models.forEach((model) => {
        const active = model.index === currentIndex;
        model.railButton.classList.toggle('is-current', active);
        model.railButton.classList.toggle('is-past', model.index < currentIndex);
        if (active) model.railButton.setAttribute('aria-current', 'page');
        else model.railButton.removeAttribute('aria-current');
      });
    });
  };

  models.forEach((model) => {
    const link = document.createElement('a');
    link.href = `#${model.id}`;
    link.textContent = `${displayNumber(model.index)} ${model.short}`;
    readerIndexList?.append(link);
  });

  const fitStage = () => {
    if (readerMode()) return;
    const bounds = stageViewport.getBoundingClientRect();
    const horizontalClearance = document.fullscreenElement ? 20 : 32;
    const verticalClearance = document.fullscreenElement ? 20 : 128;
    const scale = Math.min(
      (bounds.width - horizontalClearance) / 1600,
      (bounds.height - verticalClearance) / 900
    );
    document.documentElement.style.setProperty('--stage-scale', String(Math.max(0.1, scale)));
  };

  const indexFromHash = () => {
    let id = '';
    try { id = decodeURIComponent(window.location.hash.slice(1)); } catch { id = ''; }
    const found = models.findIndex((model) => model.id === id);
    return found >= 0 ? found : 0;
  };

  const render = (index, announce = true) => {
    currentIndex = Math.max(0, Math.min(index, models.length - 1));
    const current = models[currentIndex];

    models.forEach((model) => {
      const active = model.index === currentIndex;
      model.slide.classList.toggle('is-active', active);
      if (!readerMode()) model.slide.setAttribute('aria-hidden', String(!active));
    });

    previousButton.disabled = currentIndex === 0;
    nextButton.disabled = currentIndex === models.length - 1;
    paintNarrativeRail();
    if (slidePosition) {
      slidePosition.textContent = `${currentIndex + 1} / ${models.length}`;
      slidePosition.setAttribute('aria-label', `Slide ${currentIndex + 1} of ${models.length}`);
    }
    document.title = `${displayNumber(currentIndex)} · ${current.short} — ${baseTitle}`;
    if (announce) liveStatus.textContent = `Slide ${currentIndex + 1} of ${models.length}: ${current.title}`;
  };

  const goTo = (index) => {
    const nextIndex = Math.max(0, Math.min(index, models.length - 1));
    const nextHash = `#${models[nextIndex].id}`;
    if (window.location.hash === nextHash) render(nextIndex);
    else window.location.hash = nextHash;
  };

  const syncViewMode = (announce = false) => {
    const shouldRead = manualView === null ? narrowScreen.matches : manualView === 'reader';
    document.body.classList.toggle('reader-mode', shouldRead);
    viewToggle.textContent = shouldRead ? 'Present' : 'Read';
    viewToggle.setAttribute('aria-pressed', String(shouldRead));
    fullscreenButton.hidden = shouldRead || !document.fullscreenEnabled || typeof deckShell.requestFullscreen !== 'function';

    models.forEach((model) => {
      if (shouldRead) model.slide.removeAttribute('aria-hidden');
      else model.slide.setAttribute('aria-hidden', String(model.index !== currentIndex));
    });

    if (shouldRead) {
      requestAnimationFrame(() => {
        if (window.location.hash && currentIndex > 0) {
          models[currentIndex].slide.scrollIntoView({ block: 'start', behavior: reducedMotion.matches ? 'auto' : 'smooth' });
        } else {
          window.scrollTo({ top: 0, behavior: 'auto' });
        }
      });
    } else {
      window.scrollTo({ top: 0, behavior: 'auto' });
      requestAnimationFrame(fitStage);
    }

    if (announce) liveStatus.textContent = shouldRead ? 'Reading view enabled.' : 'Presentation view enabled.';
  };

  const syncFullscreenControl = () => {
    const active = Boolean(document.fullscreenElement);
    fullscreenButton.textContent = active ? 'Exit full screen' : 'Full screen';
    fullscreenButton.setAttribute('aria-pressed', String(active));
  };

  window.addEventListener('beforeprint', () => {
    restoreReaderAfterPrint = readerMode();
    document.body.classList.remove('reader-mode');
    models.forEach((model) => model.slide.removeAttribute('aria-hidden'));
  });

  window.addEventListener('afterprint', () => {
    if (restoreReaderAfterPrint) syncViewMode();
    else render(currentIndex, false);
    restoreReaderAfterPrint = false;
  });

  const hideFullscreenChrome = () => {
    const focused = document.activeElement;
    const chromeHasFocus = focused instanceof Element && Boolean(focused.closest('.deck-chrome'));
    if (!document.fullscreenElement || chromeHasFocus) return;
    deckShell.classList.remove('chrome-visible');
  };

  const revealFullscreenChrome = () => {
    if (!document.fullscreenElement) return;
    deckShell.classList.add('chrome-visible');
    window.clearTimeout(chromeTimer);
    chromeTimer = window.setTimeout(hideFullscreenChrome, 1500);
  };

  previousButton.addEventListener('click', () => goTo(currentIndex - 1));
  nextButton.addEventListener('click', () => goTo(currentIndex + 1));

  window.addEventListener('hashchange', () => {
    const nextIndex = indexFromHash();
    render(nextIndex);
    if (readerMode()) {
      models[nextIndex].slide.scrollIntoView({ block: 'start', behavior: reducedMotion.matches ? 'auto' : 'smooth' });
    }
  });

  window.addEventListener('keydown', (event) => {
    if (readerMode() || event.defaultPrevented || event.altKey || event.ctrlKey || event.metaKey || event.shiftKey) return;
    const target = event.target;
    if (target instanceof Element && target.closest('a, button, input, select, textarea, [contenteditable="true"]')) return;
    if (event.key === 'ArrowRight' || event.key === 'PageDown') {
      event.preventDefault();
      goTo(currentIndex + 1);
    } else if (event.key === 'ArrowLeft' || event.key === 'PageUp') {
      event.preventDefault();
      goTo(currentIndex - 1);
    }
  });

  viewToggle.addEventListener('click', () => {
    manualView = readerMode() ? 'audience' : 'reader';
    syncViewMode(true);
    render(currentIndex, false);
  });

  narrowScreen.addEventListener('change', () => {
    if (manualView === null) syncViewMode(false);
  });

  fullscreenButton.addEventListener('click', async () => {
    try {
      if (document.fullscreenElement) await document.exitFullscreen();
      else {
        fullscreenTrigger = fullscreenButton;
        await deckShell.requestFullscreen();
      }
    } catch {
      liveStatus.textContent = 'Fullscreen is unavailable. Presentation view remains active.';
    }
  });

  document.addEventListener('fullscreenchange', () => {
    syncFullscreenControl();
    window.clearTimeout(chromeTimer);
    if (document.fullscreenElement) {
      revealFullscreenChrome();
      stageViewport.focus({ preventScroll: true });
    } else {
      deckShell.classList.remove('chrome-visible');
      fullscreenTrigger?.focus({ preventScroll: true });
      fullscreenTrigger = null;
    }
    requestAnimationFrame(fitStage);
  });

  deckShell.addEventListener('pointermove', revealFullscreenChrome);
  deckShell.addEventListener('touchstart', revealFullscreenChrome, { passive: true });
  deckShell.addEventListener('focusin', (event) => {
    if (!document.fullscreenElement || !(event.target instanceof Element) || !event.target.closest('.deck-chrome')) return;
    window.clearTimeout(chromeTimer);
    deckShell.classList.add('chrome-visible');
  });
  deckShell.addEventListener('focusout', () => {
    if (document.fullscreenElement) revealFullscreenChrome();
  });

  window.addEventListener('resize', fitStage);
  document.fonts?.ready.then(fitStage);

  buildNarrativeRail();
  if (!window.location.hash) history.replaceState(null, '', `#${models[0].id}`);
  currentIndex = indexFromHash();
  render(currentIndex, false);
  syncViewMode(false);
  syncFullscreenControl();
  fitStage();
})();
