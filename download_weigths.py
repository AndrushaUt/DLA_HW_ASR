import click
import gdown

MODEL_URL = "https://drive.google.com/file/d/1jLCiwLhzg8naBenRoc_iyPrPQJUt8GmQ/view?usp=sharing"


@click.command()
@click.option("--model-url", type=str)
def main(model_url):
    model_url = model_url or MODEL_URL
    gdown.download(model_url, "best_model.pth", fuzzy=True)


if __name__ == "__main__":
    main()
