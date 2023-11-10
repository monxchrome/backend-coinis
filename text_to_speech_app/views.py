from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
import whisper
import os
from tempfile import NamedTemporaryFile

model = whisper.load_model("base")

@csrf_exempt
@require_POST
def transcribe_audio(request):
    try:
        audio_file = request.FILES['audio']

        with NamedTemporaryFile(delete=False) as temp_audio_file:
            for chunk in audio_file.chunks():
                temp_audio_file.write(chunk)

        temp_audio_file_path = temp_audio_file.name

        audio = whisper.load_audio(temp_audio_file_path)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).to(model.device)

        _, probs = model.detect_language(mel)
        detected_language = max(probs, key=probs.get)

        options = whisper.DecodingOptions(fp16 = False)
        result = whisper.decode(model, mel, options)

        response_data = {
            "detected_language": detected_language,
            "recognized_text": result.text,
        }

        os.remove(temp_audio_file_path)

        return JsonResponse(response_data)

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=400)
