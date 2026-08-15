/* The Generate button's waiting state.
 *
 * The POST behind it runs the whole correction loop and can take minutes. This
 * does one job: make the press visible immediately, and stop a second press
 * from spending the calls twice.
 */
(function () {
  if (window.__ihkamGenerateWired) return;
  window.__ihkamGenerateWired = true;

  document.addEventListener("submit", function (event) {
    var form = event.target.closest ? event.target.closest("[data-generate]") : null;
    if (!form) return;

    var button = form.querySelector("[data-generate-button]");
    var label = form.querySelector("[data-generate-label]");
    var spinner = form.querySelector("[data-generate-spinner]");
    var note = form.querySelector("[data-generate-note]");

    if (form.dataset.generateRunning === "1") {
      event.preventDefault();
      return;
    }
    form.dataset.generateRunning = "1";

    if (label) label.textContent = "Generating…";
    if (spinner) spinner.hidden = false;
    if (note) note.hidden = false;
    if (button) {
      button.setAttribute("aria-busy", "true");
      // Disabled buttons are not submitted, and the value is not needed here —
      // but the disable has to happen after the browser has taken the submit.
      window.setTimeout(function () {
        button.disabled = true;
      }, 0);
    }
  });
})();
