// Collapsible functionality for theme buttons
document.addEventListener('DOMContentLoaded', function() {
    // Handle old theme collapsible buttons
    var coll = document.getElementsByClassName("collapsible");
    var i;

    for (i = 0; i < coll.length; i++) {
        coll[i].addEventListener("click", function() {
            this.classList.toggle("active");
            var content = this.nextElementSibling;
            if (content.style.maxHeight){
                content.style.maxHeight = null;
            } else {
                content.style.maxHeight = content.scrollHeight + "px";
            }
        });
    }
});

// Modern collapsible functionality 
function toggleCollapsible(button) {
    const collapsible = button.parentElement;
    const content = collapsible.querySelector('.modern-collapsible-content');
    const icon = button.querySelector('.modern-collapsible-icon');
    
    collapsible.classList.toggle('active');
    
    if (collapsible.classList.contains('active')) {
        content.style.maxHeight = content.scrollHeight + "px";
        icon.textContent = '▲';
    } else {
        content.style.maxHeight = null;
        icon.textContent = '▼';
    }
} 