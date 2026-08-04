/* Client-Side Utilities for Student Performance Analytics */

document.addEventListener('DOMContentLoaded', () => {
  // Handle Data Cleaning triggers
  const cleanDataBtns = document.querySelectorAll('.clean-data-trigger, #clean-data-btn');
  cleanDataBtns.forEach(btn => {
    btn.addEventListener('click', async () => {
      btn.disabled = true;
      const originalHtml = btn.innerHTML;
      btn.innerHTML = '⏳ Cleaning Data...';

      try {
        const response = await fetch('/api/clean-data', { method: 'POST' });
        const result = await response.json();

        if (result.success) {
          alert(result.message);
          window.location.reload();
        } else {
          alert('Error: ' + result.message);
        }
      } catch (err) {
        alert('Data cleaning failed: ' + err.message);
      } finally {
        btn.disabled = false;
        btn.innerHTML = originalHtml;
      }
    });
  });

  // Handle CSV Upload Form
  const uploadForm = document.getElementById('upload-csv-form');
  if (uploadForm) {
    uploadForm.addEventListener('submit', async (e) => {
      e.preventDefault();
      const formData = new FormData(uploadForm);
      const submitBtn = uploadForm.querySelector('button[type="submit"]');

      submitBtn.disabled = true;
      submitBtn.innerHTML = 'Uploading & Importing...';

      try {
        const response = await fetch('/api/upload-csv', {
          method: 'POST',
          body: formData
        });
        const result = await response.json();

        if (result.success) {
          alert(result.message);
          window.location.reload();
        } else {
          alert('Upload failed: ' + result.message);
        }
      } catch (err) {
        alert('Network error: ' + err.message);
      } finally {
        submitBtn.disabled = false;
        submitBtn.innerHTML = '<i class="fa-solid fa-upload"></i> Upload & Import to ODS';
      }
    });
  }
});
