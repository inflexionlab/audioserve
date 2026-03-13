"""CLI entry point for AudioServe."""

from __future__ import annotations

import logging

import click


@click.group()
@click.option("--log-level", default="INFO", type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]))
def main(log_level: str) -> None:
    """AudioServe — Optimized inference server for audio models."""
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _parse_model_specs(model_specs: tuple[str, ...], default_dtype: str) -> list[dict]:
    """Parse model specs like 'openai/whisper-tiny' or 'facebook/wav2vec2-base-960h:float32'.

    Format: MODEL_ID[:DTYPE]
    """
    result = []
    for spec in model_specs:
        if ":" in spec:
            model_id, dtype = spec.rsplit(":", 1)
            if dtype not in ("float16", "float32", "int8", "int8_float16"):
                # Not a dtype separator — treat whole string as model ID
                result.append({"model": spec, "dtype": default_dtype})
            else:
                result.append({"model": model_id, "dtype": dtype})
        else:
            result.append({"model": spec, "dtype": default_dtype})
    return result


@main.command()
@click.option(
    "--model", "-m", multiple=True, required=True,
    help="Model ID, repeatable (e.g. -m openai/whisper-tiny -m facebook/wav2vec2-base-960h:float32)",
)
@click.option("--host", default="0.0.0.0", help="Server host")
@click.option("--port", "-p", default=8000, type=int, help="Server port")
@click.option("--dtype", default="float16", type=click.Choice(["float16", "float32", "int8", "int8_float16"]),
              help="Default dtype for models (overridden by per-model :dtype suffix)")
@click.option("--max-batch-size", default=32, type=int, help="Maximum batch size")
@click.option("--grpc-port", default=50051, type=int, help="gRPC port (0 to disable)")
@click.option("--diarization/--no-diarization", default=False, help="Enable speaker diarization")
@click.option("--hf-token", default=None, help="HuggingFace token (required for pyannote)")
def serve(
    model: tuple[str, ...],
    host: str,
    port: int,
    dtype: str,
    max_batch_size: int,
    grpc_port: int,
    diarization: bool,
    hf_token: str | None,
) -> None:
    """Start the AudioServe inference server.

    Load one or more models simultaneously:

        audioserve serve -m openai/whisper-large-v3

        audioserve serve -m openai/whisper-tiny -m facebook/wav2vec2-base-960h:float32
    """
    from audioserve.engine import AudioServeEngine

    model_specs = _parse_model_specs(model, default_dtype=dtype)

    if len(model_specs) == 1:
        engine = AudioServeEngine(
            model=model_specs[0]["model"],
            dtype=model_specs[0]["dtype"],
            max_batch_size=max_batch_size,
            diarization=diarization,
            hf_token=hf_token,
        )
    else:
        for spec in model_specs:
            spec["max_batch_size"] = max_batch_size
        engine = AudioServeEngine(
            models=model_specs,
            max_batch_size=max_batch_size,
            diarization=diarization,
            hf_token=hf_token,
        )

    engine.start()
    engine.serve(host=host, port=port, grpc_port=grpc_port or None)


@main.command()
@click.argument("audio_path")
@click.option("--model", "-m", default="openai/whisper-large-v3", help="Model ID")
@click.option("--language", "-l", default=None, help="Language code")
@click.option("--dtype", default="float16")
@click.option("--diarization/--no-diarization", default=False)
@click.option("--hf-token", default=None)
def transcribe(
    audio_path: str,
    model: str,
    language: str | None,
    dtype: str,
    diarization: bool,
    hf_token: str | None,
) -> None:
    """Transcribe an audio file locally (no server)."""
    import asyncio
    from audioserve.engine import AudioServeEngine

    engine = AudioServeEngine(model=model, dtype=dtype, diarization=diarization, hf_token=hf_token)
    engine.start()

    async def _run():
        if diarization:
            result = await engine.transcribe_with_diarization(
                audio_path, language=language
            )
            click.echo(f"\nSpeakers: {', '.join(result.speakers)}")
        else:
            result = await engine.transcribe(audio_path, language=language)

        click.echo(f"\n{result.text}")
        click.echo(f"\n[Duration: {result.duration:.1f}s | Processing: {result.processing_time:.2f}s]")

    asyncio.run(_run())
    engine.stop()


if __name__ == "__main__":
    main()
