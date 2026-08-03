/* Client-Side Utilities for Student Performance Analytics */

document.addEventListener('DOMContentLoaded', () => {
  // Handle Data Cleaning trigger
  const cleanDataBtn = document.getElementById('clean-data-btn');
  if (cleanDataBtn) {
    cleanDataBtn.addEventListener('click', async () => {
      cleanDataBtn.disabled = true;
      cleanDataBtn.innerHTML = '⏳ Cleaning Data...';

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
        cleanDataBtn.disabled = false;
        cleanDataBtn.innerHTML = '<i class="fa-solid fa-wand-magic-sparkles"></i> Clean Missing Data';
      }
    });
  }

  // Handle CSV Upload Form
  const uploadForm = document.getElementById('upload-csv-form');
  if (uploadForm) {
    uploadForm.addEventListener('submit', async (e) => {
      e.preventDefault();
      const formData = new FormData(uploadForm);
      const submitBtn = uploadForm.querySelector('button[type="submit"]');

      submitBtn.disabled = true;
      submitBtn.innerHTML = 'Uploading & Retraining...';

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
        submitBtn.innerHTML = '<i class="fa-solid fa-upload"></i> Upload & Retrain Model';
      }
    });
  }

  // Handle Clear Transaction Log
  const clearHistoryBtn = document.getElementById('clear-history-btn');
  if (clearHistoryBtn) {
    clearHistoryBtn.addEventListener('click', async () => {
      if (!confirm('Are you sure you want to clear all prediction history?')) return;

      try {
        const response = await fetch('/api/prediction-overview', { method: 'DELETE' });
        const result = await response.json();

        if (result.success) {
          alert(result.message);
          window.location.reload();
        }
      } catch (err) {
        alert('Error clearing history: ' + err.message);
      }
    });
  }

  // Navigation & Client Utilities initialized
});
