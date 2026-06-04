// Smooth page animation
document.addEventListener("DOMContentLoaded", () => {
    document.body.style.opacity = 0;

    setTimeout(() => {
        document.body.style.transition = "opacity 1s";
        document.body.style.opacity = 1;
    }, 100);
});


// Button loading animation
const forms = document.querySelectorAll("form");

forms.forEach(form => {
    form.addEventListener("submit", () => {

        let btn = form.querySelector("button");

        if(btn){
            btn.innerText = "Processing...";
            btn.style.background = "#888";
        }
    });
});


// Input highlight effect
const inputs = document.querySelectorAll("input, select");

inputs.forEach(input => {

    input.addEventListener("focus", () => {
        input.style.border = "2px solid #2c5364";
    });

    input.addEventListener("blur", () => {
        input.style.border = "1px solid #ccc";
    });

});


// Simple validation
function validateForm() {

    let screen = document.querySelector("input[name='screen']").value;

    if(screen > 20){
        alert("Screen time too high! Enter realistic value.");
        return false;
    }

    return true;
}