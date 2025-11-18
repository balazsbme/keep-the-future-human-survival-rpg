<script>
document.addEventListener('DOMContentLoaded', function () {
  const container = document.querySelector('.state-container');
  if (!container) {
    return;
  }
  const timeTarget = document.getElementById('time-status');
  const findRoot = function () {
    return container.querySelector('#state');
  };
  let root = findRoot();
  const readVersion = function () {
    return root ? parseInt(root.getAttribute('data-state-version') || '0', 10) : 0;
  };
  const readPending = function () {
    return !!(root && root.getAttribute('data-assessment-pending') === 'true');
  };
  let version = readVersion();
  let pending = readPending();
  let timer = null;
  let redirected = false;
  const shouldRedirectToVictory = function (payload) {
    if (redirected || window.location.pathname === '/start') {
      return false;
    }
    if (!payload || typeof payload !== 'object') {
      return false;
    }
    if (payload.assessment_pending) {
      return false;
    }
    if (
      typeof payload.final_score === 'number' &&
      typeof payload.win_threshold === 'number'
    ) {
      return payload.final_score >= payload.win_threshold;
    }
    return false;
  };
  const applyHtml = function (html) {
    container.innerHTML = html;
    root = findRoot();
    version = readVersion();
    pending = readPending();
  };
  const schedule = function (delay) {
    if (timer) {
      window.clearTimeout(timer);
    }
    timer = window.setTimeout(poll, delay);
  };
  const poll = function () {
    fetch('/state', { headers: { Accept: 'application/json' } })
      .then(function (response) {
        if (!response.ok) {
          throw new Error('Bad response');
        }
        return response.json();
      })
      .then(function (data) {
        if (typeof data.state_html === 'string') {
          const changedVersion =
            typeof data.progress_version === 'number' && data.progress_version !== version;
          const changedPending =
            typeof data.assessment_pending === 'boolean' && data.assessment_pending !== pending;
          if (changedVersion || changedPending || !root) {
            applyHtml(data.state_html);
          }
        }
        if (shouldRedirectToVictory(data)) {
          redirected = true;
          window.location.href = '/start';
          return;
        }
        if (typeof data.progress_version === 'number') {
          version = data.progress_version;
        }
        if (typeof data.assessment_pending === 'boolean') {
          pending = data.assessment_pending;
          if (root) {
            root.setAttribute('data-assessment-pending', pending ? 'true' : 'false');
          }
        }
        if (timeTarget && typeof data.time_status === 'string') {
          timeTarget.textContent = data.time_status;
        }
        schedule(pending ? 1200 : 4500);
      })
      .catch(function (error) {
        console.error('State polling failed', error);
        schedule(6000);
      });
  };
  schedule(pending ? 1200 : 4500);
});
</script>
