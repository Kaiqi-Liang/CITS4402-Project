const form = document.forms.range;
form.addEventListener('submit', async (event) => {
	event.preventDefault();
	const img = document.getElementById('img');
	img.style.display = 'block';
	img.src = 'spinner.svg';
	const folder = form.folder.value;
	const start = form.start.value;
	const end = form.end.value;
	const res = await fetch('http://127.0.0.1:5000', {
		method: 'POST',
		headers: {
			'Content-Type': 'application/json'
		},
		body: JSON.stringify({
			folder,
			start,
			end,
		}),
	});
	const data = await res.json();
	if (!res.ok) {
		alert(data.message)
	} else {
		img.src = '../binary.jpg';
	}
});