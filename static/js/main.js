document.addEventListener('DOMContentLoaded', function() {
    console.log('JS is working.')
    const form = document.getElementById('recommendation-form');
    const recommendationsDiv = document.getElementById('recommendations');

    form.addEventListener('submit', function(e) {
        e.preventDefault();
        const formData = new FormData(form);
        fetch('/recommend', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            recommendationsDiv.innerHTML = `<p>${data.message}</p>`;
        })
        .catch(error => console.error('Error:', error));
    });
});