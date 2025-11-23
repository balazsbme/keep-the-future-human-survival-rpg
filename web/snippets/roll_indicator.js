<script>
document.addEventListener('DOMContentLoaded', function () {
  const indicator = document.getElementById('roll-indicator');
  if (!indicator) {
    return;
  }
  const DISPLAY_DURATION = 2000;
  let hideTimer = null;
  let finalizeTimer = null;
  const scheduleHide = function () {
    indicator.classList.add('fade-out');
    hideTimer = null;
    finalizeTimer = window.setTimeout(function () {
      indicator.classList.remove('visible');
      indicator.classList.remove('fade-out');
      indicator.setAttribute('aria-hidden', 'true');
    }, 350);
  };
  const showIndicator = function () {
    indicator.classList.remove('fade-out');
    indicator.classList.add('visible');
    indicator.setAttribute('aria-hidden', 'false');
    if (hideTimer) {
      window.clearTimeout(hideTimer);
    }
    if (finalizeTimer) {
      window.clearTimeout(finalizeTimer);
    }
    hideTimer = window.setTimeout(scheduleHide, DISPLAY_DURATION);
  };
  const delaySubmitWithIndicator = function (form, predicate) {
    form.addEventListener('submit', function (event) {
      if (predicate && !predicate()) {
        return;
      }
      if (form.dataset.rollSubmitting === 'true') {
        return;
      }
      event.preventDefault();
      form.dataset.rollSubmitting = 'true';
      showIndicator();
      window.setTimeout(function () {
        form.submit();
      }, DISPLAY_DURATION);
    });
  };
  document.querySelectorAll('form.roll-trigger').forEach(function (form) {
    delaySubmitWithIndicator(form, function () { return true; });
  });
  document.querySelectorAll('form.options-form').forEach(function (form) {
    delaySubmitWithIndicator(form, function () {
      const selected = form.querySelector("input[name='response']:checked");
      return Boolean(selected && selected.dataset && selected.dataset.kind === 'action');
    });
  });
});
</script>
