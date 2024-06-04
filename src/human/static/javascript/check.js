// Function to update and display the timer
function submitCheckQuestions() {
    error = document.getElementById('end-check-error-message');

    var data = validateAnswers();
    console.log('data:', data);

    if (data != null) {
        console.log("data was not null, so submitting questions...");
        error.textContent = '';
        // Send chosen answers to server-side flask
        fetch('/submit_questions', {
            method: 'POST',
            body: JSON.stringify({
                data: data,
            }),
            headers: {
                'Content-Type': 'application/json'
            }
        })
            .then(response => response.text())
            .then(data => {
                data = JSON.parse(data);
                if (data.status == 'Success') {
                    window.location.href = '/end';
                } else {
                    console.log('error parsing submitted questions');
                }
            });
    } else {
        error.innerHTML = 'Please answer all questions.';
    }

}

function validateAnswers() {
    // Validate answers chosen by user, and if they are invalid, update error message
    // If valid, return data

    // Get q1 by name, check if it is checked
    q1 = $('input[name="question1"]')
    q2 = $('input[name="question2"]')
    q3 = $('input[name="question3"]')

    // check if the textbox chatboxComment is empty
    chatboxComment = document.getElementById('chatboxComment');

    if ((q1.is(":checked") == false) || (q2.is(":checked") == false) || (q3.is(":checked") == false) || (chatboxComment.value == '')) {
        console.log('error');
        // error.style.display = 'block';
        return null;
    } else {
        console.log('no error!');
        // Get current user selection
        // TODO: redundant, use q1/q2/q3 above
        const data = {
            'q1': document.querySelector('input[name="question1"]:checked').value,
            'q2': document.querySelector('input[name="question2"]:checked').value,
            'q3': document.querySelector('input[name="question3"]:checked').value,
            'q4': chatboxComment.value,
        };
        return data;
    }
}
