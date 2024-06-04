// Define an array of page paths
const pages = ['/', '/task1', '/task2', '/task3', '/task4', '/task5', '/chat', '/check', '/end'];
let chatMessages = [];

// Find the index of the current page in the array
function getCurrentPageIndex() {
  const currentPage = window.location.pathname;
  return pages.indexOf(currentPage);
}

// Get the progress dots container
const progressDotsContainer = document.getElementById('progress-dots');

// Define the total number of steps and the current step
const totalSteps = pages.length; // Adjust as needed
let currentStep = getCurrentPageIndex() + 1; // Change this dynamically based on the page or step

// Function to update progress dots
function updateProgressDots() {
  progressDotsContainer.innerHTML = ''; // Clear existing dots

  for (let i = 1; i <= totalSteps; i++) {
    const dot = document.createElement('div');
    dot.classList.add('progress-dot');

    // Highlight the current step
    if (i <= currentStep) {
      dot.classList.add('active');
    }

    progressDotsContainer.appendChild(dot);
  }
}

// Call the function to update progress dots on page load or as needed
updateProgressDots();


// Add event listeners to "Next" and "Previous" buttons
const nextButton = document.getElementById('next-button');
const prevButton = document.getElementById('prev-button');

function goToNextPage() {
  const currentPageIndex = getCurrentPageIndex();
  const nextPageIndex = currentPageIndex + 1;

  // Check if there is a next page
  if (nextPageIndex < pages.length) {
    // Navigate to the next page
    window.location.href = pages[nextPageIndex];
  };
}

if (nextButton) {
  nextButton.addEventListener('click', () => {
    goToNextPage();
  });
};

if (prevButton) {
  prevButton.addEventListener('click', () => {
    const currentPageIndex = getCurrentPageIndex();
    const prevPageIndex = currentPageIndex - 1;

    console.log('currentPageIndx', currentPageIndex);
    console.log('prevPageIndex', prevPageIndex);
    console.log('chat index', pages.indexOf('/chat'));

    // Check if there is a previous page
    if (prevPageIndex >= 0) {
      // Navigate to the previous page
      window.location.href = pages[prevPageIndex];
    };

  });
};



const timer = document.getElementById('timer');
if (!!timer) {
  // Call updateTimer every second to update the displayed timer
  setInterval(updateTimer, 1000);
} else {
  console.log('Timer does not exist');
};

// Function to open the modal
function openGuessReminder(currMinutes) {
  document.getElementById("reminder-to-guess").style.display = "block";
  document.getElementById("overlay").style.display = "block";
  reminder = `<center>You have ${currMinutes} minutes left. You may wish to update your guess of what <span class='math'>wug</span> does.<br><button onclick="closeGuessReminder()" style="margin-top: 30px;">OK</button></center>`
  document.getElementById("reminder-to-guess").innerHTML = reminder;
}

// Function to close the modal
function closeGuessReminder() {
  document.getElementById("reminder-to-guess").style.display = "none";
  document.getElementById("overlay").style.display = "none";
}

// Function to update and display the timer
function updateTimer() {
  var startedLearning;
  // Get started_learning value
  fetch('/get_started_learning')
    .then(response => response.text())
    .then(data => {
      startedLearning = (JSON.parse(data).data == true);
      if (!startedLearning) {
        return;
      };
      fetch('/get_all_time')
        .then(response => response.text())
        .then(data => {
          data = JSON.parse(data);
          // TODO: rename, currMinutes = minutes remaining
          currMinutes = data.minutes;
          currSeconds = data.seconds;
          formattedTime = data.formatted;
          maxMinutes = data.max_minutes;
          // Make timer visible
          document.getElementById('timer').textContent = 'Timer: ' + formattedTime;
          // If the time is divisible by 60, make the 'reminder-to-guess' div visible
          if (currSeconds == 0 && currMinutes < maxMinutes && currMinutes % 2 == 0) {
            openGuessReminder(currMinutes);
          };

        })
        .catch(error => {
          console.error('Error fetching timer:', error);
        });



      fetch('/get_raw_timer')
        .then(response => response.text())
        .then(data => {
          data = parseInt(data);
          // If reached max time, go to end page
          if (data <= 0) {
            window.location.href = '/check';
          };
        })
        .catch(error => {
          console.error('Error fetching raw timer:', error);
        });
    });
}

function saveHTML(divID) {
  // Get the HTML content of the div
  var divContent = document.getElementById(divID).innerHTML;

  // Save the HTML content in local storage
  localStorage.setItem(divID + '-html', divContent);
}

function restoreHTML(divID) {
  // Retrieve the saved HTML content from local storage
  var savedHTML = localStorage.getItem(divID + '-html');

  if (savedHTML) {
    // Set the HTML content of the div to the saved content
    document.getElementById(divID).innerHTML = savedHTML;
  }
}
