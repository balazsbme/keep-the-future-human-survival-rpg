<script>
document.addEventListener('DOMContentLoaded', function () {
  const indicator = document.getElementById('roll-indicator');
  if (!indicator) {
    return;
  }
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
    hideTimer = window.setTimeout(scheduleHide, 3200);
  };
  document.querySelectorAll('form.roll-trigger').forEach(function (form) {
    form.addEventListener('submit', showIndicator);
  });
  document.querySelectorAll('form.options-form').forEach(function (form) {
    form.addEventListener('submit', function () {
      const selected = form.querySelector("input[name='response']:checked");
      if (selected && selected.dataset && selected.dataset.kind === 'action') {
        showIndicator();
      }
    });
  });
});
</script>
