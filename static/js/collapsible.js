// Collapsible functionality for theme buttons
document.addEventListener('DOMContentLoaded', function() {
    // Handle old theme collapsible buttons
    var coll = document.getElementsByClassName("collapsible");
    var i;

    for (i = 0; i < coll.length; i++) {
        coll[i].addEventListener("click", function() {
            console.log("Button clicked"); // Debug log
            this.classList.toggle("active");
            
            // Find the next content div
            var content = this.nextElementSibling;
            
            // If nextElementSibling is not a content div, look for the next one
            while (content && !content.classList.contains("content")) {
                content = content.nextElementSibling;
            }
            
            if (content && content.classList.contains("content")) {
                console.log("Found content div"); // Debug log
                if (content.style.maxHeight && content.style.maxHeight !== "0px") {
                    content.style.maxHeight = "0px";
                } else {
                    content.style.maxHeight = content.scrollHeight + "px";
                }
            } else {
                console.log("Content div not found"); // Debug log
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