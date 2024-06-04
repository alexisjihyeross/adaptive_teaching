// Global variable to hold the status of whether or not the bot is typing.
// TODO: reset every time the page is loaded? need to fix this, but for now seems to working bc the if statements check if is_bot_typing is True
var is_bot_typing;
// Used in pullMessages to figure out if already pulled messages
var loaded_messages_before_reload = false;

const streakMinimum = 10;

function addMessageToChat(message) {
    $('#chat-list').append(message);

    var objDiv = document.getElementById("chatbox");
    objDiv.scrollTop = objDiv.scrollHeight;
}

function goToCheck() {
    window.location.href = '/check';
}

function sendUserInput() {
    // Get the value entered by the user.
    let userInput = document.getElementById('user-input').value;

    // If sending user input, shouldn't load any more messages
    loaded_messages_before_reload = true;

    // Get a reference to the error message element.
    const errorMessageElement = document.getElementById('chat-error-message');

    // Check if started_learning is true, and if not, change error message
    // Get started_learning
    fetch('/get_user_message_data')
        .then(response => response.text())
        .then(data => {
            data = JSON.parse(data).data;
            startedLearning = (data).startedLearning;
            reachedMaxPossibleInputs = (data).reachedMaxPossibleInputs;
            if (startedLearning != true) {
                errorMessageElement.innerHTML = 'You may only send messages after learning starts. Please press "Begin" to start learning.'
                return;
            }
            // Define your list of valid inputs or validation rules here.
            const validInputs = ["undefined"];


            isInteger = /^(-?[1-9]\d*|0)$/.test(userInput)
            isValid = (isInteger) || (validInputs.includes(userInput))

            is_bot_typing = getTyping();
            if (is_bot_typing) {
                errorMessageElement.innerHTML = "Please wait for the teacher to message.";
            } else if (reachedMaxPossibleInputs == true) {
                errorMessageElement.innerHTML = "You have gone through all the teacher's examples.";
            } else if (isValid) { // Check if the user's input is valid.
                // Clear the error message and proceed with submission.
                errorMessageElement.textContent = '';

                // Display the user's message in the chat.
                let chatItem = ('<li class="user-li"><div class="user-msg">' + userInput + '</div><small class="user-name">You</small></li>');
                fetch('/update_chat_user', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        // Add any additional headers if needed
                    },
                    // You can pass data in the body of the request
                    body: JSON.stringify({
                        message: userInput,
                        html: chatItem,
                        'javasscriptTime': Date.now(),
                        // Your data here, if any
                    }),
                })
                    .then(response => response.json())
                    .then(data => {
                        addMessageToChat(chatItem);

                        // Call the server-side function to get the model's response.
                        fetchModelResponse(userInput);
                    })
                    .catch(error => {
                        console.error('Error:', error);
                    });

            } else {
                // Display an error message near the input box.
                errorMessageElement.innerHTML = "Invalid input. You can provide integers or <b>undefined</b>.";
            }
        });
}

// Function that shows the Finish button (div id #finish-button)
function showFinishButton() {
    document.getElementById('finish-button').style.display = 'inline';
}

function hideFinishButton() {
    document.getElementById('finish-button').style.display = 'none';
}

function getMessageDelay(data) {
    teacherTime = data.time;
    // If the teacher's time is < MIN_DELAY, add a delay for the difference before going on to next
    MIN_DELAY = 1000;
    if (teacherTime < MIN_DELAY) {
        delay = MIN_DELAY - teacherTime;
    }
    else {
        delay = 0;
    }
    return delay;
}

function fetchModelResponse(userInput) {
    setTyping(true);

    // If getting another model response, shouldn't load any more messages
    loaded_messages_before_reload = true;

    // $('#typing-status').html('Typing...');
    var objDiv = document.getElementById("chatbox");
    objDiv.scrollTop = objDiv.scrollHeight;

    // Simulate a delay to mimic the bot thinking.
    // setTimeout(function () {
    // Replace this URL with a URL to your Flask route which gets the bot's response.
    var requestURL = '/get_response?input=' + userInput;

    fetch(requestURL)
        .then(response => response.json())
        .then(data => {
            delay = getMessageDelay(data);
            setTimeout(function () {

                let html = data.sub_html;
                // Append the bot's response to the chat list
                let chatItem = ('<li class="bot-li"><small class="bot-name">Teacher</small><div class="bot-msg">' + html + '</div></li>');
                addMessageToChat(chatItem);
                setTyping(false);

                updateStreak(data['streak']);
                updateAccuracy(data['accuracy'], data['num_correct'], data['num_total']);
                // Clean the input area
                document.getElementById('user-input').value = '';

                // Get get_user_message_data and check if reachedMaxPossibleInputs is true
                fetch('/get_user_message_data')
                    .then(response => response.json())
                    .then(data => {
                        reachedMaxPossibleInputs = (data).reachedMaxPossibleInputs;
                        if ((reachedMaxPossibleInputs) == true) {
                            showFinishButton();
                            // call update_early_finish with the appropriate reason, and wait for response
                        }
                    })
                    .catch((error) => {
                        console.error('Error:', error);
                    });
            }, delay);
        })
        .catch(error => {
            console.error('Error:', error);
        });
}

// Function that sets the button-begin-outer-wrapper to be invisible
function hideButtonBegin() {
    document.getElementById('begin-button-outer-wrapper').innerHTML = '';
    document.getElementById('begin-button-outer-wrapper').style.height = '0px';
    document.getElementById('begin-button-outer-wrapper').style.marginTop = '0px';
}

function showButtonBegin() {
    html = '<center><div id="begin-button-wrapper" class="margin-top: 0%; height: 0px;"><span>Press <b>"I Am Ready"</b> when you are ready to start learning. After that, the <b>timer</b> will begin, and you will have <b>10 minutes</b> to interact with the teacher and submit guesses. You are encouraged to play around with the interface before beginning.</span><br><br><button id="begin-button" onclick="startLearning();">I Am Ready</button></div></center>';
    document.getElementById('begin-button-outer-wrapper').innerHTML = html;
    document.getElementById('begin-button-outer-wrapper').style.height = '15vh';
    document.getElementById('begin-button-outer-wrapper').style.verticalAlign = 'center';
    document.getElementById('begin-button-outer-wrapper').style.paddingLeft = '3%';
    document.getElementById('begin-button-outer-wrapper').style.paddingRight = '3%';
    document.getElementById('begin-button-outer-wrapper').style.marginTop = '20%';
}

document.addEventListener('DOMContentLoaded', function () {
    // loop through div IDS
    var divIDs = ['result', 'guess-error-message1', 'guess-error-message2', 'function2-guess-display', 'function1-greater', 'function1-divisible', 'pred-accuracy', 'streak-container', 'reminder'];
    for (var i = 0; i < divIDs.length; i++) {
        // Restore the HTML content of the div
        if (['function1-greater', 'function1-divisible',].includes(divIDs[i])) {
            restoreCSS(divIDs[i]);
        } else {
            restoreHTML(divIDs[i]);
        }
    }

});

function saveCSS(divID) {
    element = document.getElementById(divID);
    // Use window.getComputedStyle to get the computed style of the element
    var computedStyle = window.getComputedStyle(element);

    // Convert computedStyle to a plain object
    var stylesObject = {};
    for (var i = 0; i < computedStyle.length; i++) {
        var propertyName = computedStyle[i];
        stylesObject[propertyName] = computedStyle.getPropertyValue(propertyName);
    }

    // Save the stylesObject in local storage
    localStorage.setItem(divID + '-css', JSON.stringify(stylesObject));
}

function restoreCSS(divID) {
    element = document.getElementById(divID);
    var storedStyles = localStorage.getItem(divID + '-css');
    if (storedStyles) {
        var stylesObject = JSON.parse(storedStyles);

        // Apply the styles to the element
        for (var property in stylesObject) {
            element.style[property] = stylesObject[property];
        }
    }
}


// Function to disable a dropdown option
function disableOptions(divID) {
    var dropdown = document.getElementById(divID);
    var options = dropdown.options;

    for (var i = 0; i < options.length; i++) {
        options[i].disabled = true;
    }
}

// Function to enable a dropdown option
function enableOptions(divID, excludeOptionValue) {
    var dropdown = document.getElementById(divID);
    var options = dropdown.options;

    for (var i = 0; i < options.length; i++) {
        if (options[i].value !== excludeOptionValue) {
            options[i].disabled = false;
        }
    }
}

setInterval(pullMessages, 1000);

function setTyping(val) {
    is_bot_typing = val;
    localStorage.setItem("is_bot_typing", val);
    setTypingStatus(val);
}

function setTypingStatus(is_bot_typing) {
    if (is_bot_typing) {
        $('#typing-status').html('Typing...');
    } else {
        $('#typing-status').html('');
    }

}

function getTyping() {
    typing = localStorage.getItem("is_bot_typing");
    typing = JSON.parse(typing);
    return typing;
}

// Every 2 seconds, check if there are any messages that have not been loaded
// If there are, load them
function pullMessages() {
    // Get the messages that are currently loaded by getting li elements in div #chat-list

    if (loaded_messages_before_reload) {
        return;
    }

    var loadedMessages = document.getElementById('chat-list').getElementsByTagName('li');

    if (loadedMessages.length == 0) {
        setTyping(false);
        // If there are no messages loaded, nothing to load
        return;
    }

    // Get the last message that is currently loaded
    var lastLoadedMessage = loadedMessages[loadedMessages.length - 1];

    // If sent by teacher, return
    if (lastLoadedMessage.classList.contains('bot-li')) {
        setTyping(false);
        return;
    }

    // If the last loaded message is sent by user, check if there are any additional messages to pull
    // Get last message from server
    fetch('/get_last_message')
        .then(response => response.json())
        .then(data => {
            // Check if the html of the last loaded message and the sender are the same as the last message from the server
            lastMessageHTML = data.last_message_html;
            lastMessageSender = data.last_message_sender;

            // If the last message from the server is not the same as the last loaded message AND currently what is displayed is user message, load the last message
            if (!lastMessageHTML.includes(lastLoadedMessage.innerHTML) && lastLoadedMessage.classList.contains('user-li')) {
                setTyping(true);
                // Append the last message to the chat list
                // setTyping(false);
                addMessageToChat(lastMessageHTML);
                setTyping(false);
                loaded_messages_before_reload = true;
            } else {
                setTyping(false);

            }
        })
        .catch((error) => {
            console.error('Error:', error);
        });
}

$(document).ready(function () {
    is_bot_typing = getTyping();

    setTypingStatus(is_bot_typing);

    loaded_messages_before_reload = false;

    // Get started_learning and decide whether to send message on page reload
    $.get('/get_user_message_data', function (data) {
        started_learning = (data).data.startedLearning;
        reachedMaxPossibleInputs = (data).data.reachedMaxPossibleInputs;
        if (started_learning == true) {
            // Hide the button
            hideButtonBegin();

            // Get last sender and decide whether to send message on page reload
            $.get('/get_last_sender', function (data) {
                last_sender = data.data.last_sender;
                num_messages = data.data.num_messages;
                if (last_sender != 'teacher') {
                    if (num_messages == 0) {
                        fetchModelStartMessage();
                    }
                }
            });

            enableOptions('function1', '--');
            enableOptions('function1-divisible', '--');
            enableOptions('function1-greater', '--');
            enableOptions('function2-a-calc', '--');
            enableOptions('function2-b-calc', '--');
        }
        else {
            showButtonBegin();
            disableOptions('function1');
            disableOptions('function1-divisible');
            disableOptions('function1-greater');
            disableOptions('function2-a');
            disableOptions('function2-b');
            disableOptions('function2-a-calc');
            disableOptions('function2-b-calc');

            updateStreak(0);
        }

        // Get streak by calling get_streak
        fetch('/get_streak')
            .then(response => response.json())
            .then(data => {
                streakIsMaximum = parseInt((data).streak) == streakMinimum;
                if (reachedMaxPossibleInputs) {
                    reasonForEarlyFinish = 'reached_max_possible_inputs';
                }
                else if (streakIsMaximum) {
                    reasonForEarlyFinish = 'streak_is_maximum';
                };
                if (reachedMaxPossibleInputs == true || streakIsMaximum) {
                    showFinishButton();
                    // call update_early_finish with the appropriate reason, and wait for response
                    fetch('/update_early_finish', {
                        method: 'POST',
                        body: JSON.stringify({ reason: reasonForEarlyFinish }),
                        headers: {
                            'Content-Type': 'application/json'
                        }
                    })
                        .then(response => response.json())
                        .then(data => {
                            console.log('early finish data:', data);
                        })
                        .catch((error) => {
                            console.error('Error:', error);
                        });
                }
                else {
                    hideFinishButton();
                }
            })
            .catch((error) => {
                console.error('Error:', error);
            });
    });

    fetch('/update_visited_chat', {
        method: 'POST',
        body: JSON.stringify({ message: true }),
        headers: {
            'Content-Type': 'application/json'
        }
    })
        .then(response => response.json())
        .then(data => {
            console.log('updated visited chat:', data);
        })
        .catch((error) => {
            console.error('Error:', error);
        });
});

// Function to fetching the model's response from the server.
function fetchModelStartMessage() {
    setTyping(true);

    // Simulate a delay to mimic the bot thinking.
    setTimeout(function () {
        // Replace this URL with a URL to your Flask route which gets the bot's response.
        var requestURL = '/get_start_message';

        fetch(requestURL)
            .then(response => response.json())
            .then(data => {
                let html = data.sub_html;

                // Append the bot's response to the chat list
                let chatItem = ('<li class="bot-li"><small class="bot-name">Teacher</small><div class="bot-msg">' + html + '</div></li>');
                // setTyping(false);
                addMessageToChat(chatItem);
                setTyping(false);
                document.getElementById('user-input').value = '';
            })
            .catch((error) => {
                console.error('Error:', error);
            });

    }, 1000);

}

function submitGuessFunction1(changedID) {
    const guessErrorMessageElement = document.getElementById('guess-error-message1');
    // Check if started_learning is true, if not, don't allow user to enter guess
    $.get('/get_started_learning', function (data) {
        started_learning = data.data;
        if (started_learning != true) {
            // Show error message
            guessErrorMessageElement.innerHTML = 'You must start learning before you can enter a guess. Press "I Am Ready" to start.';
            // TODO: also log the guess?

            if (true) {
                fetch('/update_guess', {
                    method: 'POST',
                    body: JSON.stringify({
                        function1: "unchanged",
                        function1Divisible: "unchanged",
                        function1Greater: "unchanged",
                        function2A: "unchanged",
                        function2B: "unchanged",
                        changed: null,
                        guessIsValid: false,
                        changedID: null, // the id of the element that was changed
                        metaData: "learning_hasn't_started",
                    }),
                    headers: {
                        'Content-Type': 'application/json'
                    }
                })
                    .then(response => response.json())
                    .then(data => {
                        console.log('updated guess:', data);
                    })
                    .catch((error) => {
                        console.error('Error:', error);
                    });
            };

        } else {
            const function1 = document.getElementById('function1').value;
            function1Divisible = null;
            function1Greater = null;

            function1IsEmpty = true;

            // This is the default val for dropdown bars in chat.html
            defaultVal = '--'

            if (function1 == 'divisible by __') {
                function1Divisible = document.getElementById('function1-divisible').value
                function1IsValid = (function1Divisible != defaultVal)
                function1_guess = function1.replace('__', '<span class="numerical-guess">' + function1Divisible + '</span>');
                function1IsEmpty = false;
            } else if (function1 == 'greater than __') {
                function1Greater = document.getElementById('function1-greater').value
                function1_guess = function1.replace('__', '<span class="numerical-guess">' + function1Greater + '</span>');
                function1IsValid = (function1Greater != defaultVal)
                function1IsEmpty = false;
                console.log('function1Greater:', function1Greater);
            } else if (function1 != defaultVal) {
                function1_guess = function1.replace('__', '<span class="numerical-guess">' + document.getElementById('function1').value + '</span>');
                function1IsValid = true;
                function1IsEmpty = false;
            } else {
                console.log(function1);
                function1IsValid = false;
                function1IsEmpty = true;
            }

            // function1IsValid and function2IsValid check if each of those guesses are complete
            if (!function1IsValid) {
                guessIsValid = false;
                guessValidErrorMessage = 'Guess for (1) is incomplete.'
                console.log('setting guessIsValid = False because function1Empty and invalid');
                guessErrorMessageElement.innerHTML = guessValidErrorMessage;
            } else {
                guessErrorMessageElement.innerHTML = "";
            }

            // remove condition
            if (true) {
                console.log('updating guess...');
                console.log('function1:', function1);
                console.log('function1Divisible:', function1Divisible);
                console.log('function1Greater:', function1Greater);
                console.log('function1IsValid:', function1IsValid);
                fetch('/update_guess', {
                    method: 'POST',
                    body: JSON.stringify({
                        function1: function1,
                        function1Divisible: function1Divisible,
                        function1Greater: function1Greater,
                        function2A: "unchanged",
                        function2B: "unchanged",
                        changed: 'function1',
                        guessIsValid: function1IsValid,
                        changedID: changedID, // the id of the element that was changed
                        metaData: null,
                    }),
                    headers: {
                        'Content-Type': 'application/json'
                    }
                })
                    .then(response => response.json())
                    .then(data => {
                        console.log('updated guess:', data);
                    })
                    .catch((error) => {
                        console.error('Error:', error);
                    });
            };

        }
        saveHTML('guess-error-message1');
    });
}

function submitGuessFunction2A() {
    const function2A = document.getElementById('function2-a').value;

    a = parseInt(function2A);

    // if a is not null
    if (a != null) {
        setSelectValue('function2-a-calc', a);
    }
    submitGuessFunction2();
    calculate();
}

function submitGuessFunction2B() {
    const function2B = document.getElementById('function2-b').value;

    b = parseInt(function2B);

    // if a is not null
    if (b != null) {
        setSelectValue('function2-b-calc', b);
    }
    submitGuessFunction2();
    calculate();
}

function submitGuessFunction2(changedID) {
    console.log("submitting Guess 2...");
    const guessErrorMessageElement = document.getElementById('guess-error-message2');
    // Check if started_learning is true, if not, don't allow user to enter guess
    $.get('/get_started_learning', function (data) {
        started_learning = data.data;
        if (started_learning != true) {
            // Show error message
            guessErrorMessageElement.innerHTML = 'You must start learning before you can enter a guess. Press "I Am Ready" to start.';
            // End function
            if (true) {
                console.log('updating guess...');
                fetch('/update_guess', {
                    method: 'POST',
                    body: JSON.stringify({
                        function1: "unchanged",
                        function1Divisible: "unchanged",
                        function1Greater: "unchanged",
                        function2A: "unchanged",
                        function2B: "unchanged",
                        changed: null,
                        guessIsValid: false,
                        changedID: null, // the id of the element that was changed
                        metaData: "learning_hasn't_started",
                    }),
                    headers: {
                        'Content-Type': 'application/json'
                    }
                })
                    .then(response => response.json())
                    .then(data => {
                        console.log('updated guess:', data);
                    })
                    .catch((error) => {
                        console.error('Error:', error);
                    });
            };

        } else {
            const function2A = document.getElementById('function2-a').value;
            const function2B = document.getElementById('function2-b').value;

            // This is the default val for dropdown bars in chat.html
            defaultVal = '--'

            console.log('function2A:', function2A)
            console.log('function2A:', typeof (function2A))
            console.log('function2B:', function2B)

            // TODO: hacky, but currently setting default value for 2A/2B to be 'null'
            if (function2A == defaultVal && function2B == defaultVal) {
                function2IsEmpty = true;
            } else {
                function2IsEmpty = false;
            }

            function2IsValid = (function2A != defaultVal) && (function2B != defaultVal)

            if (!function2IsValid) {
                guessIsValid = false;
                guessValidErrorMessage = 'Guess for (2) is incomplete.'
                console.log('setting guessIsValid = False because function2Empty and invalid');
                guessErrorMessageElement.innerHTML = guessValidErrorMessage;
            } else {
                guessErrorMessageElement.innerHTML = "";
            }

            if (true) {
                fetch('/update_guess', {
                    method: 'POST',
                    body: JSON.stringify({
                        function1: "unchanged",
                        function1Divisible: "unchanged",
                        function1Greater: "unchanged",
                        function2A: function2A,
                        function2B: function2B,
                        // TODO: also log what specific thing changed?
                        changed: 'function2',
                        guessIsValid: function2IsValid,
                        changedID: changedID, // the id of the element that was changed
                        metaData: null,
                        // guessHTML: guessHTML,
                    }),
                    headers: {
                        'Content-Type': 'application/json'
                    }
                })
                    .then(response => response.json())
                    .then(data => {
                        console.log('updated guess:', data);
                    }
                    )
                    .catch((error) => {
                        console.error('Error:', error);
                    }
                    );
            };
        }
        saveHTML('guess-error-message2');
    });
}

function startLearning() {
    // call update_started_learning
    fetch('/update_started_learning', {
        method: 'POST',
        body: JSON.stringify({
            startLearning: true,
        }),
        headers: {
            'Content-Type': 'application/json'
        }
    });

    hideButtonBegin();
    enableOptions('function1', '--');
    enableOptions('function1-divisible', '--');
    enableOptions('function1-greater', '--');
    enableOptions('function2-a', '--');
    enableOptions('function2-b', '--');
    enableOptions('function2-a-calc', '--');
    enableOptions('function2-b-calc', '--');

    setTyping(true);
    fetchModelStartMessage();
}

function changeOptions() {
    const function1 = document.getElementById('function1').value;
    reminder = 'Remember: <b>Multiple guesses</b> are allowed, and you can get <b>partial credit</b> for getting only one part of wug correct. The sooner you have the correct guess for wug, the higher your bonus.'
    invisibleDivisibleNote = "<span style='visibility: hidden;'>Note: Negative numbers can be divisible by positive numbers (i.e. <span class='math-smaller'>-4</span> is divisible by <span class='math-smaller'>2</span>). Zero is also divisible by every non-zero integer (i.e. <span class='math-smaller'>0</span> is divisible by <span class='math-smaller'>-4</span> but not by <span class='math-smaller'>0</span>). </span>";

    mathLine = "<span style='visibility: hidden; '><span class='math-smaller'>1</span></span>"

    invisiblePrimeNote = "<span style='visibility: hidden;'>Note: A prime number has only 2 factors: <span class='math-smaller'>1</span> and itself. <span class='math-smaller'>0</span>, <span class='math-smaller'>1</span>, and negative numbers are <b>not</b> prime.</span>" + "<br>" + mathLine + "<br>";
    visiblePrimeNote = "<span style='color: #888'>Note: A prime number has only 2 factors: <span class='math-smaller'>1</span> and itself. <span class='math-smaller'>0</span>, <span class='math-smaller'>1</span>, and negative numbers are <b>not</b> prime.</span>" + '<br>' + mathLine + '<br>' + mathLine;

    invisiblePositiveNote = "<span style='visibility: hidden;'>Note: 0 is <b>not</b> positive.</span>";
    visiblePositiveNote = "<span style='color: #888;'>Note: <span class='math-smaller'>0</span> is <b>not</b> positive.</span>" + "<br>" + mathLine + "<br>" + mathLine;

    if (function1 == 'divisible by __') {
        divisibleNote = "<span style='color: #888;'>Note: Negative numbers can be divisible by positive numbers (i.e. <span class='math-smaller'>-4</span> is divisible by <span class='math-smaller'>2</span>). Zero is also divisible by every non-zero integer (i.e. <span class='math-smaller'>0</span> is divisible by <span class='math-smaller'>-4</span> but not by <span class='math-smaller'>0</span>). </span>";
        fullReminder = divisibleNote + '<br><br>' + reminder
        document.getElementById("reminder").innerHTML = fullReminder;
        $("#function1-divisible").show();
        $("#function1-greater").hide();
        var selectElement = document.getElementById("function1-divisible");
        selectElement.selectedIndex = 0; // Set to the index of the default option
    } else if (function1 == 'greater than __') {
        $("#function1-greater").show();
        $("#function1-divisible").hide();
        fullReminder = invisibleDivisibleNote + '<br><br>' + reminder
        document.getElementById("reminder").innerHTML = fullReminder;
        var selectElement = document.getElementById("function1-greater");
        selectElement.selectedIndex = 0; // Set to the index of the default option
    } else {
        $("#function1-divisible").hide();
        $("#function1-greater").hide();

        if (function1 == 'prime') {
            fullReminder = visiblePrimeNote + '<br><br><br>' + reminder
        } else if (function1 == 'positive') {
            fullReminder = visiblePositiveNote + '<br><br><br><br><br>' + reminder
        } else {
            fullReminder = invisibleDivisibleNote + '<br><br>' + reminder
        }
        document.getElementById("reminder").innerHTML = fullReminder;

    }
    saveCSS('function1-divisible');
    saveCSS('function1-greater');
    saveHTML('reminder')
};

function getFormattedSpans(function2A, function2B) {
    // Returns formatted spans but without underlines, and using 'b'/'a' if values are null

    defaultVal = '--';
    /* check if not null */
    if ((function2A) !== defaultVal) {
        function2ASpan = `${function2A}`
        console.log('function2A is not null', function2A);
    } else {
        console.log('function2A is null', function2A);
        function2ASpan = `<span class='constant'>a</span>`
    }

    /* check if not null */
    if (function2B !== defaultVal) {
        function2BSpanInt = parseInt(function2B)
        if (function2BSpanInt < 0) {
            function2BSpan = `${function2B}`
        }
        else {
            function2BSpan = `+${function2B}`
        }
    } else {
        function2BSpan = `+<span class='constant'>b</span>`
    }

    return [function2ASpan, function2BSpan]
}

function getFormattedExpression(function2A, function2B) {
    // Returns formatted spans but without underlines, and using 'b'/'a' if values are null

    var spans = getFormattedSpans(function2A, function2B);
    const function2ASpan = spans[0];
    const function2BSpan = spans[1];
    function2GuessHTML = `<span class='math' style="margin-left: 5px;">${function2ASpan}&lowast;x${function2BSpan}</span>`;

    return function2GuessHTML;
}

function getFormattedExpressionUnderlined(function2A, function2B) {
    // Returns formatted spans but with underlines, and using 'b'/'a' if values are null

    var spans = getFormattedSpans(function2A, function2B);
    var function2ASpan = spans[0];
    var function2BSpan = spans[1];

    if (parseInt(function2B) < 0) {
        // TODO: hacky, but if negative, make sure negative is on outside
        function2BSpan = `-<span class='numerical-guess'>${String(function2B).substring(1)}</span>`
    }

    else {
        function2BSpan = `+<span class='numerical-guess'>${String(function2B)}</span>`
    }

    function2GuessHTML = `<span class='math' style="margin-left: 5px;"><span class='numerical-guess'>${function2ASpan}</span>&lowast;x${function2BSpan}</span>`;
    return function2GuessHTML;
}

function changeFunction2() {
    console.log('changing function2...');
    const function2A = document.getElementById('function2-a').value;
    const function2B = document.getElementById('function2-b').value;

    var function2GuessHTML = getFormattedExpression(function2A, function2B);

    $('#function2-guess-display').html(function2GuessHTML);
    saveHTML('function2-guess-display');
};

function checkEnter(event) {
    if (event.key === "Enter") {
        sendUserInput();
    }
};

function checkEnterCalculator(event) {
    if (event.key === "Enter") {
        event.preventDefault();
        calculate();
    }
};

function calculate() {
    defaultVal = '--'

    const x = parseInt(parseFloat(document.getElementById("x").value));
    const a = parseInt(document.getElementById('function2-a-calc').value);
    const b = parseInt(document.getElementById('function2-b-calc').value);

    console.log("x:", x, "a:", a, "b:", b)

    if (!isNaN(x) && !isNaN(a) && !isNaN(b)) {
        const result = a * x + b;
        var function2GuessHTML = getFormattedExpressionUnderlined(a, b);

        resultHTML = ('Result: ' + function2GuessHTML + '=' + '<span class="math result">' + result + '</span>');
        $('#result').html(resultHTML);
        localStorage.setItem('#result-html', resultHTML);
        saveHTML('result');

        document.getElementById("calculator-error-message").textContent = "";
        isValid = true;
    } else {
        document.getElementById("result").textContent = "Result:";
        // TODO: Custom error messages for all the configurations
        if (isNaN(x) && !isNaN(a) && !isNaN(b)) {
            document.getElementById("calculator-error-message").textContent = "Please enter a value for x.";
        }
        isValid = false;
    }

    // Call server-side update_calculator_usage
    var requestURL = '/update_calculator_usage';
    var request = new XMLHttpRequest();
    request.open('POST', requestURL);
    request.setRequestHeader('Content-Type', 'application/json');

    // Set a/b to be defaultVal so that they are the default values in the dropdowns
    // TODO: a bit hacky, bc parsing '--' as ints and then resetting back if null
    if (isNaN(a)) {
        a = defaultVal;
    }
    if (isNaN(b)) {
        b = defaultVal;
    }

    request.send(JSON.stringify({
        'x': x,
        'a': a,
        'b': b,
        'isValid': isValid,
    }));
}

function setSelectValue(selectId, targetValue) {
    const select = document.getElementById(selectId);
    if (select) {
        for (let i = 0; i < select.options.length; i++) {
            // TODO: need both parseInt and ===? also apply === to other checking equality
            if (parseInt(select.options[i].value) === targetValue) {
                select.selectedIndex = i;
                break;
            }
        }
    }
}

function updateCalculatorConstants() {
    var requestURL = '/get_current_guess';

    fetch(requestURL)
        .then(response => response.text())
        .then(data => {
            console.log("Updating calculator constants...")
            if (data != 'no_guess') {
                data = JSON.parse(data);
                a = parseInt(data['function2A']);
                b = parseInt(data['function2B']);

                // if a is not null
                if (a != null) {
                    console.log('Setting a to ', a)
                    setSelectValue('function2-a-calc', a);
                }
                else {
                    setSelectValue('function2-a-calc', '--')
                }
                if (b != null) {
                    console.log('Setting b to ', b)
                    setSelectValue('function2-b-calc', b);
                }
                else {
                    setSelectValue('function2-b-calc', '--')
                }
                // Set the values, and try to calculate
                calculate();
            } else {
                console.log("data is null");
                document.getElementById("calculator-error-message").textContent = "No current guess to use.";
            }
        });

}

// Function to open the modal
function openFinishConfirmation() {
    document.getElementById("finish-confirmation").style.display = "block";
    document.getElementById("finish-confirmation-overlay").style.display = "block";
    reminder = "Are you sure you want to finish? You will <b>no longer</b> be able to submit guesses for <span class='wug'>wug</span>."
    document.getElementById("finish-confirmation-text").innerHTML = reminder;

    // Add event listener to close the modal when clicking outside
    document.getElementById("finish-confirmation-overlay").addEventListener("click", closeFinishConfirmation);
}

// Function to close the modal
function closeFinishConfirmation() {
    document.getElementById("finish-confirmation").style.display = "none";
    document.getElementById("finish-confirmation-overlay").style.display = "none";

    // Remove the event listener after closing the modal
    document.getElementById("finish-confirmation-overlay").removeEventListener("click", closeFinishConfirmation);
}

window.addEventListener('beforeunload', function (e) {
    // Check if a specific button is clicked
    var buttonClicked = document.getElementById('finish-button-yes');

    if (buttonClicked) {
        // Do not prevent the default behavior if the button is clicked
        return;
    }

    // Cancel the event
    e.preventDefault();
    // Standard-compliant browsers
    e.returnValue = '';
    // For older browsers
    return '';
});

function updateStreak(correctStreak) {
    const streakContainer = document.getElementById('streak-container');
    streakContainer.innerHTML = 'Streak: ';

    for (let i = 0; i < streakMinimum; i++) {
        const streakItem = document.createElement('div');
        streakItem.classList.add('streak-item', i < correctStreak ? 'correct-pred' : 'incorrect-pred');
        streakItem.innerText = i + 1;
        streakContainer.appendChild(streakItem);
    }
    saveHTML('streak-container');

    // If the streak is streakMinimum, show finish button
    if (correctStreak == streakMinimum) {
        showFinishButton();
        // call update_early_finish with the appropriate reason, and wait for response
        reasonForEarlyFinish = 'streak_is_minimum';
        // call update_early_finish
        fetch('/update_early_finish', {
            method: 'POST',
            body: JSON.stringify({ reason: reasonForEarlyFinish }),
            headers: {
                'Content-Type': 'application/json'
            }
        })
            .then(response => response.json())
            .then(data => {
                console.log('early finish data:', data);
            })
            .catch((error) => {
                console.error('Error:', error);
            });
    }
}

function updateAccuracy(accuracy, numCorrect, numTotal) {
    const accuracyElement = document.getElementById('pred-accuracy');
    accuracyElement.innerText = `${numCorrect}/${numTotal}`;
    saveHTML('pred-accuracy');
}
