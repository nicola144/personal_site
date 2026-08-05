document.addEventListener('DOMContentLoaded', function () {
  var filters = document.querySelectorAll('.pub-filter');
  var items = document.querySelectorAll('.pub-list .pub-item');
  if (!filters.length || !items.length) return;

  filters.forEach(function (btn) {
    btn.addEventListener('click', function () {
      var group = btn.getAttribute('data-filter') || 'all';

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
  });
});
