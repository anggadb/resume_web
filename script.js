const mobileMenuBtn = document.getElementById('mobile-menu-btn');
const mobileMenu = document.getElementById('mobile-menu');
const hamburgerIcon = document.getElementById('hamburger-icon');
const closeIcon = document.getElementById('close-icon');
const mobileLinks = document.querySelectorAll('.mobile-link');

function toggleMenu() {
    const isOpen = !mobileMenu.classList.contains('hidden');
    if (isOpen) {
        mobileMenu.classList.add('hidden');
        hamburgerIcon.classList.remove('hidden');
        closeIcon.classList.add('hidden');
    } else {
        mobileMenu.classList.remove('hidden');
        hamburgerIcon.classList.add('hidden');
        closeIcon.classList.remove('hidden');
    }
}

mobileMenuBtn.addEventListener('click', toggleMenu);
mobileLinks.forEach(link => link.addEventListener('click', toggleMenu));

const aiForm = document.getElementById('ai-form');
const aiInput = document.getElementById('ai-input');
const responseBox = document.getElementById('ai-response-box');
const responseText = document.getElementById('ai-response-text');

aiForm.addEventListener('submit', async (e) => {
    e.preventDefault();

    const question = aiInput.value.trim();
    if (!question) return;

    responseBox.classList.remove('hidden');
    responseText.innerHTML = `<span class="text-slate-400 typing-indicator">Analyzing Your Question</span>`;

    try {
        await new Promise(resolve => setTimeout(resolve, 1500));

        responseText.innerHTML = `Answer: <strong>"${question}"</strong>. <br><br>Ganti bagian JavaScript ini dengan perintah <code>fetch('/api/chat', ...)</code> saat backend AI Anda sudah aktif di Vercel atau server lain.`;

    } catch (error) {
        responseText.innerHTML = `<span class="text-red-400">Terjadi kesalahan koneksi. Silakan coba lagi.</span>`;
    }
});