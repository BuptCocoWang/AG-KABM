### README

To train a model, run the following command in the current directory:

```bash
python3 -u ./model/train.py
```

This will use the data from the `./data` directory to train a model and save it to the `./pth` directory. The training and validation datasets are split in a 7:3 ratio.

To test the model, run the following command:

```bash
python3 -u ./model/test.py ${model_name.pth}
```

This will use the data from the `./test_data` directory for testing.
