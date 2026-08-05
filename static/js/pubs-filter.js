// Event delegation so filters survive footer math rewriting .markdown innerHTML.
document.addEventListener('click', function (event) {
  var btn = event.target.closest('.pub-filter');
  if (!btn || !btn.closest('.pub-filters')) return;

  var group = btn.getAttribute('data-filter') || 'all';
  var filters = document.querySelectorAll('.pub-filter');
  var items = document.querySelectorAll('.pub-list .pub-item');

  filters.forEach(function (b) {
    b.classList.toggle('is-active', b === btn);
  });

  items.forEach(function (item) {
    if (group === 'all') {
      item.classList.remove('is-hidden');
      return;
    }
    var groups = (item.getAttribute('data-groups') || '').split(/\s+/);
    item.classList.toggle('is-hidden', groups.indexOf(group) === -1);
  });
});
