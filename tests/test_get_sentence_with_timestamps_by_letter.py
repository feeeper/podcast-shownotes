import json

from src.shownotes import (
    get_shownotes_with_timestamps,
    get_topic_texts,
    get_sentences_with_timestamps,
    get_sentences_with_timestamps_by_letter
)
from src.models import Transcription, Segment, Shownotes

from stanza import Pipeline

pipeline = Pipeline('ru', processors=['tokenize'])


def test_get_sentences_with_timestamps_simple_case():
    segments_json = [
        {"id":0,"seek":0,"start":0,"end":5.42,"text":" Всем привет, и с вами DevZen Podcast.","tokens":[],"temperature":0,"avg_logprob":-0.43,"compression_ratio":1.801,"no_speech_prob":0.379},
        {"id":1,"seek":0,"start":5.42,"end":8.22,"text":" Сегодня замечательный выпуск.","tokens":[],"temperature":0,"avg_logprob":-0.437,"compression_ratio":1.801,"no_speech_prob":0.379},
        {"id":2,"seek":0,"start":8.22,"end":10.22,"text":" Постапокалиптический выпуск.","tokens":[],"temperature":0,"avg_logprob":-0.437,"compression_ratio":1.801,"no_speech_prob":0.379},
        {"id":3,"seek":0,"start":10.22,"end":15.22,"text":" Представьте себя, три человека сидят при свечках, рядом","tokens":[],"temperature":0,"avg_logprob":-0.437,"compression_ratio":1.801,"no_speech_prob":0.379},
        {"id":4,"seek":0,"start":15.22,"end":21.02,"text":" с светящимся в темноте экраном мониторы лэптопа","tokens":[],"temperature":0,"avg_logprob":-0.437,"compression_ratio":1.801,"no_speech_prob":0.379},
        {"id":5,"seek":0,"start":21.02,"end":22.740000000000002,"text":" и пытаются записывать пока.","tokens":[],"temperature":0,"avg_logprob":-0.437,"compression_ratio":1.801,"no_speech_prob":0.379}]
    transcription = Transcription(
        text=' Всем привет, и с вами DevZen Podcast. Сегодня замечательный выпуск. Постапокалиптический выпуск. Представьте себя, три человека сидят при свечках, рядом с светящимся в темноте экраном мониторы лэптопа и пытаются записывать пока.',
        segments=[Segment(**x) for x in segments_json],
        language='ru')

    pipeline = Pipeline('ru', processors=['tokenize'])

    sentences_with_timestamps = get_sentences_with_timestamps_by_letter(transcription, lambda text: [x.text for x in pipeline(text).sentences])

    assert len(sentences_with_timestamps) == 4
    assert 0 == sentences_with_timestamps[0][0] and 5.42 == sentences_with_timestamps[0][1] and "Всем привет, и с вами DevZen Podcast." == sentences_with_timestamps[0][2]
    assert 5.42 == sentences_with_timestamps[1][0] and 8.22 == sentences_with_timestamps[1][1] and "Сегодня замечательный выпуск." == sentences_with_timestamps[1][2]
    assert 8.22 == sentences_with_timestamps[2][0] and 10.22 == sentences_with_timestamps[2][1] and "Постапокалиптический выпуск." == sentences_with_timestamps[2][2]
    assert 10.22 == sentences_with_timestamps[3][0] and 22.740000000000002 == sentences_with_timestamps[3][1] and "Представьте себя, три человека сидят при свечках, рядом с светящимся в темноте экраном мониторы лэптопа и пытаются записывать пока." == sentences_with_timestamps[3][2]


# awful case for stanza.
# Stanza counts segments 1253, 1254, 1255, 1256 as
# a [["То есть это не 100%"], ["А, не 100% С"], ["и не 100% отсутствие П."], ["Вот это так можно расценить."]]
# but in whole episode transcript splits to [
#  'То есть это не на 100%.',
#  'То есть это не 100% А, не 100%',
#  'С и не 100% отсутствие П. Вот это так можно расценить.'
#  ]
def test_get_sentences_with_timestamps_segment_could_not_split_into_sentences_correctly():
    segments_json = [
        {"id":1252,"seek":342192,"start":3431.92,"end":3433.92,"text":" Я понял. То есть это не на 100%.","tokens":[],"temperature":0,"avg_logprob":-0.187,"compression_ratio":2.240,"no_speech_prob":0.271},
        {"id":1253,"seek":342192,"start":3433.92,"end":3435.92,"text":" То есть это не 100%","tokens":[],"temperature":0,"avg_logprob":-0.187,"compression_ratio":2.24,"no_speech_prob":0.271},
        {"id":1254,"seek":342192,"start":3435.92,"end":3437.92,"text":" А, не 100% С","tokens":[],"temperature":0,"avg_logprob":-0.187,"compression_ratio":2.24,"no_speech_prob":0.271},
        {"id":1255,"seek":342192,"start":3437.92,"end":3439.92,"text":" и не 100% отсутствие П.","tokens":[],"temperature":0,"avg_logprob":-0.187,"compression_ratio":2.24,"no_speech_prob":0.271},
        {"id":1256,"seek":342192,"start":3439.92,"end":3441.92,"text":" Вот это так можно расценить.","tokens":[],"temperature":0,"avg_logprob":-0.187,"compression_ratio":2.24,"no_speech_prob":0.271}
    ]

    transcription = Transcription(
        text=' Я понял. То есть это не на 100%. То есть это не 100% А, не 100% С и не 100% отсутствие П. Вот это так можно расценить.',
        segments=[Segment(**x) for x in segments_json],
        language='ru')

    sentences_with_timestamps = get_sentences_with_timestamps_by_letter(
        transcription,
        lambda text: [x.text for x in pipeline(text).sentences])

    assert len(sentences_with_timestamps) == 4
    assert 3431.92 == sentences_with_timestamps[0][0] and 3433.92 == sentences_with_timestamps[0][1] and 'Я понял.' == sentences_with_timestamps[0][2]
    assert 3431.92 == sentences_with_timestamps[1][0] and 3433.92 == sentences_with_timestamps[1][1] and 'То есть это не на 100%.' == sentences_with_timestamps[1][2]
    assert 3433.92 == sentences_with_timestamps[2][0] and 3437.92 == sentences_with_timestamps[2][1] and 'То есть это не 100% А, не 100%' == sentences_with_timestamps[2][2]
    assert 3435.92 == sentences_with_timestamps[3][0] and 3441.92 == sentences_with_timestamps[3][1] and 'С и не 100% отсутствие П. Вот это так можно расценить.' == sentences_with_timestamps[3][2]


def test_get_sentences_with_timestamps_111():
    segments_json = [
        {"id":410,"seek":138440,"start":1390.4,"end":1393.4,"text":" Так как compare-and-set у нас работает нормально,","tokens":[],"temperature":0,"avg_logprob":-0.184,"compression_ratio":1.745,"no_speech_prob":0.025},
        {"id":411,"seek":138440,"start":1393.4,"end":1396.4,"text":" даже в sequential consistency, если он есть,","tokens":[],"temperature":0,"avg_logprob":-0.184,"compression_ratio":1.745,"no_speech_prob":0.025},
        {"id":412,"seek":138440,"start":1396.4,"end":1398.4,"text":" то получается sequential consistency","tokens":[],"temperature":0,"avg_logprob":-0.184,"compression_ratio":1.745,"no_speech_prob":0.025},
        {"id":413,"seek":138440,"start":1398.4,"end":1401.4,"text":" с compare-and-set дает нам linearizability.","tokens":[],"temperature":0,"avg_logprob":-0.184,"compression_ratio":1.745,"no_speech_prob":0.025},
        {"id":414,"seek":138440,"start":1401.4,"end":1403.4,"text":" И тогда мы можем,","tokens":[],"temperature":0,"avg_logprob":-0.184,"compression_ratio":1.745,"no_speech_prob":0.025},
        {"id":415,"seek":138440,"start":1403.4,"end":1406.4,"text":" используя вот этот read mapping и write mapping,","tokens":[],"temperature":0,"avg_logprob":-0.184,"compression_ratio":1.745,"no_speech_prob":0.025},
        {"id":416,"seek":138440,"start":1406.4,"end":1409.4,"text":" взять и свести задачу","tokens":[],"temperature":0,"avg_logprob":-0.184,"compression_ratio":1.745,"no_speech_prob":0.025},
        {"id":417,"seek":140940,"start":1410.4,"end":1414.4,"text":" с np.complete к nlogn.n.","tokens":[],"temperature":0,"avg_logprob":-0.271,"compression_ratio":1.982,"no_speech_prob":0.179},
        {"id":418,"seek":140940,"start":1414.4,"end":1417.4,"text":" И тогда можно вообще все протестировать очень быстро.","tokens":[],"temperature":0,"avg_logprob":-0.271,"compression_ratio":1.982,"no_speech_prob":0.179}
    ]

    transcription = Transcription(
        text=' Так как compare-and-set у нас работает нормально, даже в sequential consistency, если он есть, то получается sequential consistency с compare-and-set дает нам linearizability. И тогда мы можем, используя вот этот read mapping и write mapping, взять и свести задачу с np.complete к nlogn.n. И тогда можно вообще все протестировать очень быстро.',
        segments=[Segment(**x) for x in segments_json],
        language='ru')

    sentences_with_timestamps = get_sentences_with_timestamps_by_letter(
        transcription,
        lambda text: [x.text for x in pipeline(text).sentences],)

    assert len(sentences_with_timestamps) == 4
    assert 1390.4 == sentences_with_timestamps[0][0] and 1401.4 == sentences_with_timestamps[0][1] and 'Так как compare-and-set у нас работает нормально, даже в sequential consistency, если он есть, то получается sequential consistency с compare-and-set дает нам linearizability.' == sentences_with_timestamps[0][2]
    assert 1401.4 == sentences_with_timestamps[1][0] and 1414.4 == sentences_with_timestamps[1][1] and 'И тогда мы можем, используя вот этот read mapping и write mapping, взять и свести задачу с np.complete к nlogn.' == sentences_with_timestamps[1][2]
    assert 1410.4 == sentences_with_timestamps[2][0] and 1414.4 == sentences_with_timestamps[2][1] and 'n.' == sentences_with_timestamps[2][2]
    assert 1414.4 == sentences_with_timestamps[3][0] and 1417.4 == sentences_with_timestamps[3][1] and 'И тогда можно вообще все протестировать очень быстро.' == sentences_with_timestamps[3][2]


def test_unknown():
    segments_json = [
        {"id":480,"seek":261004,"start":2624.04,"end":2630.04,"text":" Ну вот я вначале не сказал, опять же, повторюсь, я работаю в компании Postgres Pro","tokens":[],"temperature":0,"avg_logprob":-0.229,"compression_ratio":1.739,"no_speech_prob":0.014},
        {"id":481,"seek":261004,"start":2630.04,"end":2639.04,"text":" и в основном занимаюсь Postgres, и какое-то время назад мне попала задачка поревьюить реализацию","tokens":[],"temperature":0,"avg_logprob":-0.229,"compression_ratio":1.739,"no_speech_prob":0.014},
        {"id":482,"seek":263904,"start":2639.04,"end":2645.04,"text":" такого стандарта SQL.JSON для Postgres.","tokens":[],"temperature":0,"avg_logprob":-0.248,"compression_ratio":1.795,"no_speech_prob":0.054},
        {"id":483,"seek":263904,"start":2645.04,"end":2652.04,"text":" Ну и вот какое-то время уже хотел на это внимательнее посмотреть, ну и так вот все сложилось.","tokens":[],"temperature":0,"avg_logprob":-0.248,"compression_ratio":1.795,"no_speech_prob":0.054},
    ]

    transcription = Transcription(
        text=' Ну вот я вначале не сказал, опять же, повторюсь, я работаю в компании Postgres Pro и в основном занимаюсь Postgres, и какое-то время назад мне попала задачка поревьюить реализацию такого стандарта SQL.JSON для Postgres. Ну и вот какое-то время уже хотел на это внимательнее посмотреть, ну и так вот все сложилось.',
        segments=[Segment(**x) for x in segments_json],
        language='ru')

    sentences_with_timestamps = get_sentences_with_timestamps_by_letter(
        transcription,
        lambda text: [x.text for x in pipeline(text).sentences],)

    assert len(sentences_with_timestamps) == 3
    assert 2624.04 == sentences_with_timestamps[0][0] and 2645.04 == sentences_with_timestamps[0][1] and 'Ну вот я вначале не сказал, опять же, повторюсь, я работаю в компании Postgres Pro и в основном занимаюсь Postgres, и какое-то время назад мне попала задачка поревьюить реализацию такого стандарта SQL.' == sentences_with_timestamps[0][2]
    assert 2639.04 == sentences_with_timestamps[1][0] and 2645.04 == sentences_with_timestamps[1][1] and 'JSON для Postgres.' == sentences_with_timestamps[1][2]
    assert 2645.04 == sentences_with_timestamps[2][0] and 2652.04 == sentences_with_timestamps[2][1] and 'Ну и вот какое-то время уже хотел на это внимательнее посмотреть, ну и так вот все сложилось.' == sentences_with_timestamps[2][2]


def test_get_sentences_with_timestamps_segment_contains_two_sentences():
    segments_json = [
        {"id":96,"seek":27538,"start":297.38,"end":301.38,"text":" Поэтому это был как-то пробный шар, и потом мы вышли,","tokens":[],"temperature":0,"avg_logprob":-0.313,"compression_ratio":2.079,"no_speech_prob":0.1},
        {"id":97,"seek":27538,"start":301.38,"end":304.38,"text":" собственно, отдать наизнанный проект в Apache.","tokens":[],"temperature":0,"avg_logprob":-0.313,"compression_ratio":2.079,"no_speech_prob":0.1},
        {"id":98,"seek":30438,"start":304.38,"end":307.38,"text":" Делается так. Пишется пропозал.","tokens":[],"temperature":0,"avg_logprob":-0.183,"compression_ratio":1.997,"no_speech_prob":0.041},
        {"id":99,"seek":30438,"start":307.38,"end":310.38,"text":" Пропозал, наверное, самая главная часть того,","tokens":[],"temperature":0,"avg_logprob":-0.183,"compression_ratio":1.997,"no_speech_prob":0.041},
        {"id":100,"seek":30438,"start":310.38,"end":314.38,"text":" как податься в инкубатор.","tokens":[],"temperature":0,"avg_logprob":-0.183,"compression_ratio":1.997,"no_speech_prob":0.041}
    ]
    transcription = Transcription(
        text=' Поэтому это был как-то пробный шар, и потом мы вышли, собственно, отдать наизнанный проект в Apache. Делается так. Пишется пропозал. Пропозал, наверное, самая главная часть того, как податься в инкубатор.',
        segments=[Segment(**x) for x in segments_json],
        language='ru')

    sentences_with_timestamps = get_sentences_with_timestamps_by_letter(transcription, lambda text: [x.text for x in pipeline(text).sentences])

    assert len(sentences_with_timestamps) == 4
    assert 297.38 == sentences_with_timestamps[0][0] and 304.38 == sentences_with_timestamps[0][1] and 'Поэтому это был как-то пробный шар, и потом мы вышли, собственно, отдать наизнанный проект в Apache.' == sentences_with_timestamps[0][2]
    assert 304.38 == sentences_with_timestamps[1][0] and 307.38 == sentences_with_timestamps[1][1] and 'Делается так.' == sentences_with_timestamps[1][2]
    assert 304.38 == sentences_with_timestamps[2][0] and 307.38 == sentences_with_timestamps[2][1] and 'Пишется пропозал.' == sentences_with_timestamps[2][2]
    assert 307.38 == sentences_with_timestamps[3][0] and 314.38 == sentences_with_timestamps[3][1] and 'Пропозал, наверное, самая главная часть того, как податься в инкубатор.' == sentences_with_timestamps[3][2]


def test_get_sentences_with_timestamps_segment_contains_three_sentences():
    segments_json = [
        {"id":96,"seek":27538,"start":297.38,"end":301.38,"text":" Поэтому это был как-то пробный шар, и потом мы вышли,","tokens":[],"temperature":0,"avg_logprob":-0.313,"compression_ratio":2.079,"no_speech_prob":0.1},
        {"id":97,"seek":27538,"start":301.38,"end":304.38,"text":" собственно, отдать наизнанный проект в Apache.","tokens":[],"temperature":0,"avg_logprob":-0.313,"compression_ratio":2.079,"no_speech_prob":0.1},
        {"id":98,"seek":30438,"start":304.38,"end":307.38,"text":" Делается так. Пишется пропозал. Проверяется пропозал.","tokens":[],"temperature":0,"avg_logprob":-0.183,"compression_ratio":1.997,"no_speech_prob":0.041},
        {"id":99,"seek":30438,"start":307.38,"end":310.38,"text":" Пропозал, наверное, самая главная часть того,","tokens":[],"temperature":0,"avg_logprob":-0.183,"compression_ratio":1.997,"no_speech_prob":0.041},
        {"id":100,"seek":30438,"start":310.38,"end":314.38,"text":" как податься в инкубатор.","tokens":[],"temperature":0,"avg_logprob":-0.183,"compression_ratio":1.997,"no_speech_prob":0.041}
    ]
    transcription = Transcription(
        text=' Поэтому это был как-то пробный шар, и потом мы вышли, собственно, отдать наизнанный проект в Apache. Делается так. Пишется пропозал. Проверяется пропозал. Пропозал, наверное, самая главная часть того, как податься в инкубатор.',
        segments=[Segment(**x) for x in segments_json],
        language='ru')

    sentences_with_timestamps = get_sentences_with_timestamps_by_letter(transcription, lambda text: [x.text for x in pipeline(text).sentences])

    assert len(sentences_with_timestamps) == 5
    assert 297.38 == sentences_with_timestamps[0][0] and 304.38 == sentences_with_timestamps[0][1] and 'Поэтому это был как-то пробный шар, и потом мы вышли, собственно, отдать наизнанный проект в Apache.' == sentences_with_timestamps[0][2]
    assert 304.38 == sentences_with_timestamps[1][0] and 307.38 == sentences_with_timestamps[1][1] and 'Делается так.' == sentences_with_timestamps[1][2]
    assert 304.38 == sentences_with_timestamps[2][0] and 307.38 == sentences_with_timestamps[2][1] and 'Пишется пропозал.' == sentences_with_timestamps[2][2]
    assert 304.38 == sentences_with_timestamps[3][0] and 307.38 == sentences_with_timestamps[3][1] and 'Проверяется пропозал.' == sentences_with_timestamps[3][2]
    assert 307.38 == sentences_with_timestamps[4][0] and 314.38 == sentences_with_timestamps[4][1] and 'Пропозал, наверное, самая главная часть того, как податься в инкубатор.' == sentences_with_timestamps[4][2]


def test_get_sentences_with_timestamps_segment_contains_full_prev_sent_and_beginning_of_next_sent():
    segments_json = [
        {"id":99,"seek":30438,"start":307.38,"end":310.38,"text":" Пропозал, наверное, самая главная часть того,","tokens":[],"temperature":0,"avg_logprob":-0.183,"compression_ratio":1.997,"no_speech_prob":0.041},
        {"id":100,"seek":30438,"start":310.38,"end":314.38,"text":" как податься в инкубатор.","tokens":[],"temperature":0,"avg_logprob":-0.183,"compression_ratio":1.997,"no_speech_prob":0.041},
        {"id":101,"seek":30438,"start":314.38,"end":318.38,"text":" Обычно пропозал... Там есть очень четкая структура,","tokens":[],"temperature":0,"avg_logprob":-0.183,"compression_ratio":1.997,"no_speech_prob":0.041},
        {"id":102,"seek":30438,"start":318.38,"end":321.38,"text":" что должно быть в пропозале, написать, какой, собственно,","tokens":[],"temperature":0,"avg_logprob":-0.183,"compression_ratio":1.997,"no_speech_prob":0.041},
        {"id":103,"seek":30438,"start":321.38,"end":325.38,"text":" задача проекта решает, что он даст комьюнити.","tokens":[],"temperature":0,"avg_logprob":-0.183,"compression_ratio":1.997,"no_speech_prob":0.041}
    ]
    transcription = Transcription(
        text=' Пропозал, наверное, самая главная часть того, как податься в инкубатор. Обычно пропозал... Там есть очень четкая структура, что должно быть в пропозале, написать, какой, собственно, задача проекта решает, что он даст комьюнити.',
        segments=[Segment(**x) for x in segments_json],
        language='ru')

    sentences_with_timestamps = get_sentences_with_timestamps_by_letter(
        transcription,
        lambda text: [x.text for x in pipeline(text).sentences])

    assert len(sentences_with_timestamps) == 3
    assert 307.38 == sentences_with_timestamps[0][0] and 314.38 == sentences_with_timestamps[0][1] and 'Пропозал, наверное, самая главная часть того, как податься в инкубатор.' == sentences_with_timestamps[0][2]
    assert 314.38 == sentences_with_timestamps[1][0] and 318.38 == sentences_with_timestamps[1][1] and 'Обычно пропозал...' == sentences_with_timestamps[1][2]
    assert 314.38 == sentences_with_timestamps[2][0] and 325.38 == sentences_with_timestamps[2][1] and 'Там есть очень четкая структура, что должно быть в пропозале, написать, какой, собственно, задача проекта решает, что он даст комьюнити.' == sentences_with_timestamps[2][2]


def test_get_sentences_with_timestamps_segment_text_split_single_dot():
    segments_json = [
        {"id":60,"seek":16738,"start":167.38,"end":170.38,"text":" Сейчас, примерно год назад...","tokens":[],"temperature":0,"avg_logprob":-0.273,"compression_ratio":1.903,"no_speech_prob":0.015},
        {"id":61,"seek":16738,"start":170.38,"end":174.38,"text":" Изначально этот проект составлен как чисто университетский","tokens":[],"temperature":0,"avg_logprob":-0.273,"compression_ratio":1.903,"no_speech_prob":0.015},
        {"id":62,"seek":16738,"start":174.38,"end":177.38,"text":" ресерш-проект, но потом...","tokens":[],"temperature":0,"avg_logprob":-0.273,"compression_ratio":1.903,"no_speech_prob":0.015},
        {"id":63,"seek":16738,"start":177.38,"end":178.38,"text":" Я кинул в гидр.","tokens":[],"temperature":0,"avg_logprob":-0.273,"compression_ratio":1.903,"no_speech_prob":0.015},
        {"id":64,"seek":16738,"start":178.38,"end":184.38,"text":" Потом, примерно год назад, мы вышли в Apache-инкубатор,","tokens":[],"temperature":0,"avg_logprob":-0.273,"compression_ratio":1.903,"no_speech_prob":0.015},
        {"id":65,"seek":16738,"start":184.38,"end":187.38,"text":" и где-то полгода назад это уже является топ-левел","tokens":[],"temperature":0,"avg_logprob":-0.273,"compression_ratio":1.903,"no_speech_prob":0.015},
        {"id":66,"seek":16738,"start":187.38,"end":189.38,"text":" Apache-проекта.","tokens":[],"temperature":0,"avg_logprob":-0.273,"compression_ratio":1.903,"no_speech_prob":0.015}
    ]

    transcription = Transcription(
        text=' Сейчас, примерно год назад... Изначально этот проект составлен как чисто университетский ресерш-проект, но потом... Я кинул в гидр. Потом, примерно год назад, мы вышли в Apache-инкубатор, и где-то полгода назад это уже является топ-левел Apache-проекта.',
        segments=[Segment(**x) for x in segments_json],
        language='ru')

    sentences_with_timestamps = get_sentences_with_timestamps_by_letter(
        transcription,
        lambda text: [x.text for x in pipeline(text).sentences])

    assert len(sentences_with_timestamps) == 4
    assert 167.38 == sentences_with_timestamps[0][0] and 170.38 == sentences_with_timestamps[0][1] and 'Сейчас, примерно год назад...' == sentences_with_timestamps[0][2]
    assert 170.38 == sentences_with_timestamps[1][0] and 177.38 == sentences_with_timestamps[1][1] and 'Изначально этот проект составлен как чисто университетский ресерш-проект, но потом...' == sentences_with_timestamps[1][2]
    assert 177.38 == sentences_with_timestamps[2][0] and 178.38 == sentences_with_timestamps[2][1] and 'Я кинул в гидр.' == sentences_with_timestamps[2][2]
    assert 178.38 == sentences_with_timestamps[3][0] and 189.38 == sentences_with_timestamps[3][1] and 'Потом, примерно год назад, мы вышли в Apache-инкубатор, и где-то полгода назад это уже является топ-левел Apache-проекта.' == sentences_with_timestamps[3][2]


def test_get_sentences_with_timestamps_segment_contains_end_of_prev_sent_and_full_next_sent():
    segments_json = [
        {"id":407,"seek":161738,"start":1632.38,"end":1636.38,"text":" Во-вторых, очень большой эффорт вначале был приложен на то,","tokens":[],"temperature":0,"avg_logprob":-0.228,"compression_ratio":1.793,"no_speech_prob":0.074},
        {"id":408,"seek":161738,"start":1636.38,"end":1643.38,"text":" чтобы не использовать... минимизировать эффект garbage collection в Java.","tokens":[],"temperature":0,"avg_logprob":-0.228,"compression_ratio":1.793,"no_speech_prob":0.074},
        {"id":409,"seek":164338,"start":1644.38,"end":1647.38,"text":" Для этого мы используем...","tokens":[],"temperature":0,"avg_logprob":-0.314,"compression_ratio":1.818,"no_speech_prob":0.061}
    ]

    transcription = Transcription(
        text=' Во-вторых, очень большой эффорт вначале был приложен на то, чтобы не использовать... минимизировать эффект garbage collection в Java. Для этого мы используем...',
        segments=[Segment(**x) for x in segments_json],
        language='ru')

    sentences_with_timestamps = get_sentences_with_timestamps_by_letter(
        transcription,
        lambda text: [x.text for x in pipeline(text).sentences])

    assert len(sentences_with_timestamps) == 3
    assert 1632.38 == sentences_with_timestamps[0][0] and 1643.38 == sentences_with_timestamps[0][1] and 'Во-вторых, очень большой эффорт вначале был приложен на то, чтобы не использовать...' == sentences_with_timestamps[0][2]
    assert 1636.38 == sentences_with_timestamps[1][0] and 1643.38 == sentences_with_timestamps[1][1] and 'минимизировать эффект garbage collection в Java.' == sentences_with_timestamps[1][2]
    assert 1644.38 == sentences_with_timestamps[2][0] and 1647.38 == sentences_with_timestamps[2][1] and 'Для этого мы используем...' == sentences_with_timestamps[2][2]


def test_get_sentences_with_timestamps_segment_has_new_sent_starts_with_lower_and_prev_ends_with_dots():
    segments_json = [
        {"id":1727,"seek":406800,"start":4092,"end":4094,"text":" Я смотрю на это и говорю, подождите,","tokens":[],"temperature":0,"avg_logprob":-0.17,"compression_ratio":2.038,"no_speech_prob":0.303},
        {"id":1728,"seek":406800,"start":4094,"end":4096,"text":" а у вас разве нету какого-то","tokens":[],"temperature":0,"avg_logprob":-0.17,"compression_ratio":2.038,"no_speech_prob":0.303},
        {"id":1729,"seek":409600,"start":4096,"end":4098,"text":" стандарта для диссертации","tokens":[],"temperature":0,"avg_logprob":-0.167,"compression_ratio":2.006,"no_speech_prob":0.044},
        {"id":1730,"seek":409600,"start":4098,"end":4100,"text":" там в","tokens":[],"temperature":0,"avg_logprob":-0.167,"compression_ratio":2.006,"no_speech_prob":0.044},
        {"id":1731,"seek":409600,"start":4100,"end":4102,"text":" там...темплейт где-нибудь.","tokens":[],"temperature":0,"avg_logprob":-0.167,"compression_ratio":2.006,"no_speech_prob":0.044},
    ]

    transcription = Transcription(
        text=' Я смотрю на это и говорю, подождите, а у вас разве нету какого-то стандарта для диссертации там в там...темплейт где-нибудь.',
        segments=[Segment(**x) for x in segments_json],
        language='ru')

    sentences_with_timestamps = get_sentences_with_timestamps_by_letter(
        transcription,
        lambda text: [x.text for x in pipeline(text).sentences])

    assert len(sentences_with_timestamps) == 3
    assert 4092 == sentences_with_timestamps[0][0] and 4102 == sentences_with_timestamps[0][1] and 'Я смотрю на это и говорю, подождите, а у вас разве нету какого-то стандарта для диссертации там в там..' == sentences_with_timestamps[0][2]
    assert 4100 == sentences_with_timestamps[1][0] and 4102 == sentences_with_timestamps[1][1] and '.' == sentences_with_timestamps[1][2]
    assert 4100 == sentences_with_timestamps[2][0] and 4102 == sentences_with_timestamps[2][1] and 'темплейт где-нибудь.' == sentences_with_timestamps[2][2]


def test_get_sentences_with_timestamps_segment_splits_incorrectly_into_sentences_if_contains_dots():
    segments_json = [
        {"id":1751,"seek":412400,"start":4140,"end":4142,"text":" А оказывается, можно совершенно","tokens":[],"temperature":0,"avg_logprob":-0.205,"compression_ratio":2.042,"no_speech_prob":0.115},
        {"id":1752,"seek":412400,"start":4142,"end":4144,"text":" иначе. И это, наверное, идет","tokens":[],"temperature":0,"avg_logprob":-0.205,"compression_ratio":2.042,"no_speech_prob":0.115},
        {"id":1753,"seek":412400,"start":4144,"end":4146,"text":" из нашего... нас просто","tokens":[],"temperature":0,"avg_logprob":-0.205,"compression_ratio":2.042,"no_speech_prob":0.115},
        {"id":1754,"seek":412400,"start":4146,"end":4148,"text":" готовят как хороших, прекрасных","tokens":[],"temperature":0,"avg_logprob":-0.205,"compression_ratio":2.042,"no_speech_prob":0.115},
        {"id":1755,"seek":412400,"start":4148,"end":4150,"text":" исполнителей. Вот почему","tokens":[],"temperature":0,"avg_logprob":-0.205,"compression_ratio":2.042,"no_speech_prob":0.115},
        {"id":1756,"seek":412400,"start":4150,"end":4152,"text":" белорусские, российские инженеры так сильно ценятся,","tokens":[],"temperature":0,"avg_logprob":-0.205,"compression_ratio":2.042,"no_speech_prob":0.115},
        {"id":1757,"seek":415200,"start":4152,"end":4154,"text":" когда мы прекрасно умеем исполнять.","tokens":[],"temperature":0,"avg_logprob":-0.158,"compression_ratio":1.904,"no_speech_prob":0.175}
    ]

    transcription = Transcription(
        text=' А оказывается, можно совершенно иначе. И это, наверное, идет из нашего... нас просто готовят как хороших, прекрасных исполнителей. Вот почему белорусские, российские инженеры так сильно ценятся, когда мы прекрасно умеем исполнять.',
        segments=[Segment(**x) for x in segments_json],
        language='ru')

    sentences_with_timestamps = get_sentences_with_timestamps_by_letter(
        transcription,
        lambda text: [x.text for x in pipeline(text).sentences])

    assert len(sentences_with_timestamps) == 4
    assert 4140 == sentences_with_timestamps[0][0] and 4144 == sentences_with_timestamps[0][1] and 'А оказывается, можно совершенно иначе.' == sentences_with_timestamps[0][2]
    assert 4142 == sentences_with_timestamps[1][0] and 4146 == sentences_with_timestamps[1][1] and 'И это, наверное, идет из нашего...' == sentences_with_timestamps[1][2]
    assert 4144 == sentences_with_timestamps[2][0] and 4150 == sentences_with_timestamps[2][1] and 'нас просто готовят как хороших, прекрасных исполнителей.' == sentences_with_timestamps[2][2]
    assert 4148 == sentences_with_timestamps[3][0] and 4154 == sentences_with_timestamps[3][1] and 'Вот почему белорусские, российские инженеры так сильно ценятся, когда мы прекрасно умеем исполнять.' == sentences_with_timestamps[3][2]