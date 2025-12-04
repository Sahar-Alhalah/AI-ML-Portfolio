document.addEventListener('DOMContentLoaded', function () {
    // Add focus and blur effects on input fields
    const inputs = document.querySelectorAll('input[type="number"], select');
    inputs.forEach(input => {
        input.addEventListener('focus', () => {
            input.style.borderColor = '#3a7bd5';
            input.style.boxShadow = '0 0 5px rgba(58, 123, 213, 0.5)';
        });
        input.addEventListener('blur', () => {
            input.style.borderColor = '#ddd';
            input.style.boxShadow = 'none';
        });
    });

    // Validate form inputs before submission
    const form = document.querySelector('form');
    form.addEventListener('submit', function (event) {
        let valid = true;
        inputs.forEach(input => {
            if (input.value === '' || isNaN(input.value)) {
                valid = false;
                input.style.borderColor = '#d8000c';
            } else {
                input.style.borderColor = '#ddd'; // Reset the border color if valid
            }
        });
        if (!valid) {
            event.preventDefault();
            flashMessage('Please fill in all required fields correctly.', 'danger'); // Use flash message instead
        }
    });

    function flashMessage(message, type) {
        // Create a flash message element
        const flashDiv = document.createElement('div');
        flashDiv.className = `alert alert-${type}`;
        flashDiv.textContent = message;
        document.body.insertBefore(flashDiv, form); // Insert before the form

        // Remove the flash message after a timeout
        setTimeout(() => {
            flashDiv.remove();
        }, 3000);
    }
});
