from molgen.models.merging import make_rewarded_soups
import argparse
import os

if __name__ == "__main__":

    # Load training run
    parser = argparse.ArgumentParser(prog='python create_rewarded_soups.py')
    parser.add_argument('--models', help="list of paths to the models to merge", type=str, nargs="+", required=True)
    parser.add_argument('--coefficients', help="merging coefficients", type=str, nargs="+", required=True)
    parser.add_argument('--output_path', help="path to store the merged model", type=str, required=True)
    args = parser.parse_args()

    assert len(args.coefficients) == len(args.models), "Number of coefficients must match number of models"
    args.coefficients = [float(c) for c in args.coefficients]
    for model_path in args.models: assert os.path.exists(model_path), f"Model path {model_path} does not exist"
    
    # Create merged model
    model, tokenizer = make_rewarded_soups(models=args.models, coefficients=args.coefficients)
    model.save_pretrained(args.output_path)
    tokenizer.save_pretrained(args.output_path)