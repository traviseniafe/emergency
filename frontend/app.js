// Elements on the page that we need to change from JavaScript.
const statusDot = document.getElementById("status-dot");
const statusText = document.getElementById("status-text");
const recheckButton = document.getElementById("recheck");

// Update the coloured dot and the message next to it.
// state is one of "ok", "bad" or "wait".
function setStatus(state, message) {
  statusDot.className = "dot " + state;
  statusText.textContent = message;
}

// Ask the backend whether it is alive.
async function checkBackend() {
  setStatus("wait", "Checking...");
  try {
    const response = await fetch("/api/health");
    if (!response.ok) {
      throw new Error("HTTP " + response.status);
    }
    const data = await response.json();
    setStatus("ok", "Backend online (version " + data.version + ")");
  } catch (error) {
    setStatus("bad", "Backend unreachable: " + error.message);
  }
}

recheckButton.addEventListener("click", checkBackend);
checkBackend(); // run once when the page loads