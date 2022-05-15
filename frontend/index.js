const form = document.forms.range;
form.addEventListener('submit', async (event) => {
	event.preventDefault();
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
	if (!res.ok) {
		const data = await res.json();
		alert(data.message)
	}
});