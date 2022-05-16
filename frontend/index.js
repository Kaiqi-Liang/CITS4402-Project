const form = document.forms.range;
form.addEventListener('submit', async (event) => {
	event.preventDefault();
	const spinner = document.getElementById('spinner');
	spinner.style.display = 'block';
	spinner.src = 'spinner.svg';
	const detection = document.getElementById('detection');
	const match = document.getElementById('match');
	match.style.display = 'none';
	detection.style.display = 'none';
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
	spinner.style.display = 'none';
	if (!res.ok) {
		const snackbar = document.getElementById('alert');
		snackbar.style.display = 'flex';
		snackbar.innerText = data.message;
		setTimeout(() => {
			snackbar.style.display = 'none';
		}, 2000);
	} else {
		document.getElementById('title').style.display = 'block';
		detection.src = '../candidate_detection.jpg';
		detection.style.display = 'block';
		match.src = '../region_growing.jpg';
		match.style.display = 'block';
	}
});