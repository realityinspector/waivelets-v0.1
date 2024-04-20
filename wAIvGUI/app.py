from shiny import App, reactive, render, ui
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import datetime
import json
import csv
import pywt
import asyncio
from matplotlib import transforms
from sentence_transformers import SentenceTransformer


app_ui = ui.page_sidebar(  
    ui.sidebar(
        ui.card(
          ui.card_header(ui.h4("Select a Text for Analysis")),
          ui.input_select(  
            "selectText",  
            "Select an option below:",  
            {"bookGatsby": "The Great Gatsby"}, 
          ),
          ui.input_action_button("embed_text", "Embed Text"),
          ui.output_text("run_embedding"),
          ui.input_action_button("run_analysis", "Run Analysis"),
          ui.input_action_button("show_analysis", "Show Analysis"),
        ), 
    ),
    ui.layout_columns(
      ui.card(
        ui.card_header("Wavelet Power Spectrum"),
        ui.input_slider("embed_dimension", "Embedding Dimension", 1, 384, 1),
        ui.output_plot("plot_wav_power"),
        full_screen=True,
      ),
      ui.card(
        ui.card_header("Text Under Analysis"),
        #ui.output_data_frame("latest_data"),
      ),
      col_widths=[9, 3],
    ),
  ui.output_ui("run_power_spectrum"),
)


def server(input, output, session):
  embedText = []
  powerSpectrum = []
  wavelet = 'morl'
  scales = np.arange(1, 1024)


  @render.text()
  @reactive.event(input.embed_text)
  def run_embedding():
    with open('./texts/gatsby.json', 'r') as f:
      content = json.load(f)

    model = SentenceTransformer("all-MiniLM-L6-v2")
    nonlocal embedText
    embedText = model.encode(content['para'])

    print(embedText.shape)
    print(embedText[1].shape)
    print(range(len(embedText[1])))
    return f"Embedding Complete"

  @output
  @render.ui
  @reactive.event(input.run_analysis)
  async def run_power_spectrum():

    nonlocal embedText
    nonlocal powerSpectrum
    nonlocal wavelet
    nonlocal scales

    with ui.Progress(min=min(range(len(embedText[1]))), max=max(range(len(embedText[1])))) as p:
        p.set(message="Power Spectrum Analysis in progress", detail="This may take a while...")
            
        for i in range(len(embedText[1])):
          p.set(i, message="Analysing")
          coef, freq = pywt.cwt(embedText[:, i], scales, wavelet)
          power = (abs(coef))**2
          powerSpectrum.append(power)
          await asyncio.sleep(0.1)

    return "Done Analysing!"
    
  @render.plot(alt="Wavelet Power Spectrum")  
  @reactive.event(input.show_analysis)
  def plot_wav_power():  
      nonlocal powerSpectrum
      nonlocal scales
      powerMax = np.max(powerSpectrum)
    
      fig, ax = plt.subplots()
      powerScale = ax.imshow(powerSpectrum[int(input.embed_dimension() - 1)], cmap='coolwarm', vmin=0, vmax=powerMax, aspect='auto', origin='lower', extent=[0, len(embedText[:,int(input.embed_dimension()) - 1]), 0, len(scales)])
      ax.set_title("Power Spectrum")
      ax.set_xlabel("Paragraph (html def)")
      ax.set_ylabel("Period")

      fig.colorbar(ax=ax,
         mappable=powerScale,
         label="Wavelet Power",
         location="left")
    
      return fig  
  
app = App(app_ui, server)
