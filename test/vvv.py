import wandb

api = wandb.Api()
run = api.run("predict-woo/mesh-ssm/golden-night-20")
for file in run.files():
    print(file)
    # You can also filter the file.download() based on the file name too
    # print(file)
