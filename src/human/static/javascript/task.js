$(document).ready(function () {
    // Get last sender and decide whether to send message on page reload
    document.getElementById("timer").style.display = 'none';
    $.get('/get_visited_chat', function (data) {
        visitedChat = data.data;
        if (visitedChat) {
            console.log("Visited chat, setting timer display to visible")
            document.getElementById("timer").style.display = 'block';
        }
    });
});

// Function to update and display the timer
function submitTaskQuestions() {

    error = document.getElementById('task-check-error-message');

    var data = validateAnswers();
    console.log('data:', data);

    if (data != null) {
        console.log("data was not null, so submitting questions...");
        error.textContent = '';
        // Send chosen answers to server-side flask
        fetch('/submit_task_questions', {
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
                showAnswers();
                saveHTML('task-check-error-message');
                saveHTML('question1-result');
                saveHTML('question2-result');
            });
    } else {
        console.log('data is null');
        error.innerHTML = 'Please answer all questions.';
    }

}

function validateAnswers() {
    console.log('validating answers...');
    // Validate answers chosen by user, and if they are invalid, update error message
    // If valid, return data

    // Get q1 by name, check if it is checked
    q1 = $('input[name="question1"]')
    q2 = $('input[name="question2"]')

    if ((q1.is(":checked") == false) || (q2.is(":checked") == false)) {
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
        };
        return data;
    }
}

document.addEventListener('DOMContentLoaded', function () {
    // loop through div IDS
    var divIDs = ['task-check-error-message', 'question1-result', 'question2-result'];
    // TODO: also set visibilities
    for (var i = 0; i < divIDs.length; i++) {
        console.log(localStorage);
        // Restore the HTML content of the div
        restoreHTML(divIDs[i]);
    };

});



function showAnswers() {

    // Define correct answers
    const correctAnswers = {
        question1: "both",
        question2: "true"
    };

    // Get user's answers
    const userAnswers = {
        question1: document.querySelector('input[name="question1"]:checked').value,
        question2: document.querySelector('input[name="question2"]:checked').value
    };

    question1Correctness = userAnswers.question1 === correctAnswers.question1
    if (question1Correctness) {
        question1Stub = '<span class="correct-answer">Correct!</span> '
    }
    else {
        question1Stub = '<span class="incorrect-answer">Incorrect!</span> '
    }
    document.getElementById('question1-result').innerHTML = question1Stub + 'The correct answer is <i>Both (1) and (2)</i>.';

    // The ? notation is a ternary operator, which is a shorthand for if/else
    question2Correctness = userAnswers.question2 === correctAnswers.question2
    if (question2Correctness) {
        question2Stub = '<span class="correct-answer">Correct!</span> '
    }
    else {
        question2Stub = '<span class="incorrect-answer">Incorrect!</span> '
    }
    document.getElementById('question2-result').innerHTML = question2Stub + 'The correct answer is <i>True</i>. Multiple guesses are allowed.';

}

function validateSubmitted() {
    // Call get_submitted_task_questions and validate that the instruction questions have been submitted

    fetch('/get_submitted_task_questions', {
        method: 'GET',
        headers: {
            'Content-Type': 'application/json'
        }
    })
        .then(response => response.text())
        .then(data => {
            data = JSON.parse(data);
            console.log("data:", data);
            submitted = data.is_submitted;
            console.log("submitted:", submitted);
            if (submitted) {
                console.log("Submitted, redirecting to chat");
                window.location.href = '/chat';
            } else {
                // Set next-error-message message
                console.log("Not submitted, setting next-error-message to visible")
                document.getElementById("next-error-message").innerHTML = 'Please submit answers to questions.';
            }
        });
};
