# ✅ Fine-Tuning Checklist

Use this checklist when training your model on Kaggle.

## Before Training

- [ ] Have a Kaggle account (free at kaggle.com)
- [ ] Opened `kaggle_fine_tune.py` in this project
- [ ] Read `KAGGLE_GUIDE.md` for detailed instructions

## Kaggle Setup

- [ ] Created new Kaggle Notebook
- [ ] Enabled GPU: Settings → Accelerator → "GPU T4 x2"
- [ ] Kept Internet ON (needed for dataset download)
- [ ] Copied entire `kaggle_fine_tune.py` into notebook
- [ ] Added dataset: "b-mc2/sql-create-context"

## During Training (2-3 hours)

Monitor these outputs:
- [ ] ✅ Dependencies installed successfully
- [ ] ✅ Dataset loaded (9000 train, 1000 validation)
- [ ] ✅ Model loaded with LoRA applied
- [ ] ✅ Training started (see loss decreasing)
- [ ] ✅ Validation loss < 0.3 (target)
- [ ] ✅ Model saved to `/kaggle/working/fine_tuned_text2sql_codet5`
- [ ] ✅ Test queries generated successfully

## After Training

- [ ] Training completed without errors
- [ ] Saw message: "🎉 FINE-TUNING COMPLETE!"
- [ ] Clicked "Output" tab in Kaggle
- [ ] Found `fine_tuned_text2sql_codet5/` folder
- [ ] Downloaded the folder (ZIP file)

## Local Setup

- [ ] Extracted ZIP file
- [ ] Moved to `text-to-sql-chatbot/models/fine_tuned_text2sql_codet5/`
- [ ] Verified these files exist:
  - [ ] `config.json`
  - [ ] `adapter_config.json`
  - [ ] `adapter_model.bin`
  - [ ] `tokenizer_config.json`
  - [ ] `tokenizer.json`
  - [ ] `model_info.txt`

## Testing

- [ ] Ran: `python -c "from app.model_utils import load_model; load_model()"`
- [ ] Saw: "✅ Model loaded successfully"
- [ ] Started app: `./run.sh`
- [ ] App running at http://127.0.0.1:5000
- [ ] Uploaded test CSV/Excel file
- [ ] Asked SIMPLE question (e.g., "What is the total revenue?")
- [ ] Saw in logs: "🤖 Routing to local model"
- [ ] Query returned correct results
- [ ] Asked COMPLEX question (e.g., "Show year-over-year growth")
- [ ] Saw in logs: "🚀 Routing to Groq API"

## Verification

Run these tests:
```bash
# Test classifier
python test_classifier.py
# Expected: 12/12 tests passing

# Test integration
python test_integration.py
# Expected: All 3 workflows complete
```

- [ ] Classifier tests: 12/12 passing
- [ ] Integration tests: All passing
- [ ] No errors in Flask app startup

## Success Criteria

✅ You have successfully fine-tuned your model if:

1. Model files exist in `models/fine_tuned_text2sql_codet5/`
2. Flask app starts without errors
3. Simple queries route to local model (see logs: "🤖 Routing to local model")
4. Complex queries route to Groq API (see logs: "🚀 Routing to Groq API")
5. Queries return accurate results

## Expected Performance

After fine-tuning, your system should have:

| Metric | Value |
|--------|-------|
| Simple queries accuracy | 75-85% |
| Complex queries accuracy | 90-95% |
| Overall accuracy | 85-90% |
| Offline capability | 75% |
| Avg latency (simple) | ~300ms |
| Avg latency (complex) | ~200ms |
| Effective throughput | ~120 queries/min |

## Troubleshooting

### Model not loading
**Check:**
```bash
ls -la models/fine_tuned_text2sql_codet5/
# Should show: config.json, adapter_model.bin, etc.
```

### Still routing to Groq for simple queries
**Possible causes:**
1. Model not in correct location
2. Model files corrupted during download
3. Need to restart Flask app
4. Check logs for errors

**Solution:**
```bash
# Verify model exists
python -c "import os; print('Model exists:', os.path.exists('models/fine_tuned_text2sql_codet5/config.json'))"

# Restart app
./run.sh
```

### High error rate
**Possible causes:**
1. Training didn't complete successfully
2. Validation loss was too high (>0.5)
3. Need to train longer (increase EPOCHS)

**Solution:**
- Check `model_info.txt` for validation loss
- If loss >0.5, retrain with more epochs
- Try full dataset (TRAIN_SAMPLES = None)

## Next Steps After Success

1. ✅ Test with your own datasets
2. ✅ Try different question types
3. ✅ Monitor routing decisions (check logs)
4. ✅ Share feedback or issues

## Resources

- **Training Code:** `kaggle_fine_tune.py`
- **Detailed Guide:** `KAGGLE_GUIDE.md`
- **System Docs:** `README.md`
- **Implementation:** `IMPLEMENTATION_SUMMARY.md`

---

**Happy Fine-Tuning! 🚀**

Need help? Check the troubleshooting section in `KAGGLE_GUIDE.md`
