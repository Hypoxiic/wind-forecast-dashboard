/*  assets/close_debug.js
    --------------------------------------------
    Automatically hide Dash Dev‑Tools (Callbacks /
    Errors overlay) when the user presses Escape
    or clicks outside the overlay.
    Works only when app is running in debug=True.
*/
(function () {
    // helper: send message Dash DevTools listens for
    function hideDevTools() {
        window.postMessage(
            {
                type: "__dash_devtools__",
                subType: "setState",
                payload: { isVisible: false }   // <-- key flag
            },
            "*"
        );
    }

    // ── ESC key ───────────────────────────────
    document.addEventListener("keydown", (e) => {
        if (e.key === "Escape") {
            hideDevTools();
        }
    });

    // ── click outside the inspector ───────────
    document.addEventListener("click", (e) => {
        const dbg = document.querySelector(".dash-debug-menu");
        if (dbg && !dbg.contains(e.target)) {
            hideDevTools();
        }
    });
})();
