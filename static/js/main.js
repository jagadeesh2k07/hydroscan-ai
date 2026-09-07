// main.js — shared across every page (nav toggle, toast helper)

document.addEventListener("DOMContentLoaded", () => {
  const toggle = document.getElementById("navToggle");
  const links = document.querySelector(".nav-links");
  if (toggle && links) {
    toggle.addEventListener("click", () => {
      links.classList.toggle("is-open");
    });
  }
});

function hsToast(message, ms = 2600) {
  let el = document.querySelector(".toast");
  if (!el) {
    el = document.createElement("div");
    el.className = "toast";
    document.body.appendChild(el);
  }
  el.textContent = message;
  requestAnimationFrame(() => el.classList.add("is-visible"));
  clearTimeout(el._t);
  el._t = setTimeout(() => el.classList.remove("is-visible"), ms);
}
