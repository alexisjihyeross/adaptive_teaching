// Function to tell the user that loading may take a bit after they submit their prolific id
function showLoading() {
    // Get the user entry in the form prolific-id, and if not empty/null, change error
    prolificId = document.getElementById('prolific-id-input').value;
    console.log('prolificId:', prolificId);
    // if not null
    if (prolificId != null && prolificId != '') {
        document.getElementById('loading').style.display = 'block';
        document.getElementById('loading').innerHTML = 'Loading, please wait a few moments...';
    }
}
