import os 
import pandas as pd 
import mlflow


def setup_mlflow(experiment = "Drug-LLM", tracking_uri = "file:./mlruns"):
    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.set_experiment(experiment)
    return experiment.experiment_id

def log_parameters(lora_r, lora_alpha, lora_dropout, batch_size, gradient_accumulation_steps, learning_rate, num_epochs, max_length):
    mlflow.log_param("LORA_A", lora_r)
    mlflow.log_param("LORA_ALPHA", lora_alpha)
    mlflow.log_param("LORA_DROPOUT", lora_dropout)
    mlflow.log_param("BATCH_SIZE", batch_size)
    mlflow.log_param("GRADIENT_ACCUM_STEPS", gradient_accumulation_steps)
    mlflow.log_param("LEARNING_RATE", learning_rate)
    mlflow.log_param("NUM_EPOCHS", num_epochs)
    mlflow.log_param("MAX__LENGTH", max_length)

def log_metrics(eval_loss, perplexity):
    mlflow.log_metric("eval_loss", eval_loss)
    mlflow.log_metric("perplexity", perplexity)