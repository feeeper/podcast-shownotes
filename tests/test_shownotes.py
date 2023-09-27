import json

import pytest

from src.shownotes import (
    get_shownotes_with_timestamps,
    get_topic_texts,
    get_sentences_with_timestamps,
    get_sentences_with_timestamps_by_letter
)
from src.models import Transcription, Segment, Shownotes

from stanza import Pipeline

pipeline = Pipeline('ru', processors=['tokenize'])


def test_get_shownotes_should_return_shownotes_basic():
    shownotes_text = '\n\nТемы выпуска: все, что вы хотели узнать о 2600, безопасности беспроводных сетей и том, как ограбить поезд без оружия; обмазываемся Elixir’ом: вакансия на удаленку, предстоящий релиз 1.4, митап в Rambler’е, и не только; другие предстоящие и прошедшие конференции и митапы; новости от Amazon: Aurora теперь совместима с PostgreSQL, новая альтернатива Digital Ocean, новые инстансы с FPGA; как разработчики Scylla чинили баги в ядре Linux; потрясающие новости для любителей игр; и, конечно же, темы наших слушателей. \nШоу нотес:\n\n\n[00:00] Благодарим наших замечательных патронов\n[00:02] Расспрашиваем гостей\n\nhttp://defcon.su/\nhttps://twitter.com/dc7499\n\n\n[01:07] Вакансия от компании Netronix\n\nhttp://netronixgroup.com\nhttp://bit.ly/backend-ruby\n\n\n[01:08] Эликсир 1.4 на подходе\n[01:21] Positive Hack Days 23-24 мая\n[01:26] Прошел Flink meetup Berlin\n\nhttp://www.slideshare.net/robertmetzger1/apache-flink-community-updates-november-2016-berlin-meetup\nhttp://www.slideshare.net/StefanRichter10/a-look-at-flink-12\n\n\n[01:32] Митап Elixir-разработчиков в Rambler&Co\n[01:33] Global Game Jam 2017 Berlin Registration\n[01:37] Strip Hack Moscow 2016\n[01:47] Amazon Aurora Update – PostgreSQL Compatibility\n\nhttps://aws.amazon.com/blogs/aws/amazon-aurora-update-postgresql-compatibility/\nhttps://news.ycombinator.com/item?id=13072861\n\n\n[01:52] Убийца DO от Amazon\n\nhttps://amazonlightsail.com/pricing/\nhttps://news.ycombinator.com/item?id=13072155\n\n\n[01:58] Новые инстансы EC2 с программируемым железом\n\nhttps://aws.amazon.com/blogs/aws/developer-preview-ec2-instances-f1-with-programmable-hardware/\nhttp://papilio.cc/\n\n\n[02:06] Big latencies? It’s open season for kernel bug-hunting!\n\nhttp://www.scylladb.com/2016/11/14/cfq-kernel-bug/\nhttps://github.com/iovisor/bcc\n\n\n[02:11] The Last Of Us — Part II\n[02:13] Темы и вопросы слушателей для 0121\n\n\n\nПоддержи DevZen Podcast на Patreon!\n\nЛог чата: https://gitter.im/DevZenRu/live/archives/2016/12/03\nГолоса выпуска: Светлана, Валерий, Иван, Александр а также гости — Олег и Юрий.\nФоновая музыка: Plastic3 — Corporate Rock Motivation Loop 4\n\n\n\n\nПодкаст: Скачать (80.4MB) \n\n \n'
    shownotes = get_shownotes_with_timestamps(shownotes_text)
    assert len(shownotes) == 15


def test_get_shownotes_should_return_shownotes_when_shownotes_has_duplicated_timestamps():
    shownotes_text = 'Темы выпуска: обсуждаем обработку больших данных на Asterixdb; новые тулзы для ведения тем, теперь c таймстемпами; делимся впечатления о DevZen митапе; разбираем отличную статью про то, как правильно обрабатывать слишком большую нагрузку; Саша делится впечатлениями о погружении в пайку и электронику; защищаем MacOS от разных атак; покупаем FLAC и пытаемся понять, стоит ли его вообще слушать. \nШоу нотес:\n\n\n[00:00] Расспрашиваем гостя\n\nBDMS на Java Asterixdb\n\n\n[00:45] Саша запилил скрипт для шоунотов\n\nhttps://github.com/afiskon/py-topics-storage\nhttp://ell.stackexchange.com/questions/109940/what-is-the-difference-between-theme-and-topic\n\n\n[00:48] Отчет о прошедшем митапе\n[00:53] Handling Overload\n\nhttp://ferd.ca/handling-overload.html\nhttps://twitter.com/mononcqc/status/801842170739228672\nhttps://github.com/basho/sidejob\n\n\n[01:05] В закладки — стартер кит по Ардуине\n[01:05] Про электронику, пайку и роботов\n\nhttp://eax.me/electronics-first-steps/\nhttp://www.chipdip.ru/product/arduino-mobile-robots/\nhttp://www.ozon.ru/context/detail/id/135412298/\n\n\n[01:09] Учимся работать с asyncio+aiohttp: в Москве пройдет курс от Core-разработчика Python\n\nhttps://habrahabr.ru/company/pt/blog/315818/\nhttp://asvetlov.blogspot.ru/2016/11/asyncio-aiohttp-training.html\n\n\n[01:10] В закладки — видео с прошлых ZeroNights\n\nhttps://www.youtube.com/playlist?list=PLHlFrzuFU1EU2vGbM2siqEGJ-_ogPPgoM\nhttps://www.youtube.com/playlist?list=PLHlFrzuFU1EV7ssrR4Per7y88z-tKC_K5\nhttps://soundcloud.com/konstantinnovikovofficial/zeronights-2016-welcome-performance\nhttps://play.google.com/store/music/artist?id=Amiz2hraqawhptde7baio3oochm\n\n\n[01:13] Как правильно защищать свою MacOS\n[01:15] Как сделать нормальную стартовую страницу в Chromium\n[01:19] Где можно легально купить FLAC\n\nhttp://bandcamp.com/\nhttps://bandcamp.com/afiskon\nhttps://bandcamp.com/valerymeleshkin\nhttps://www.cnet.com/news/top-6-sites-for-buying-flac-music/\n\n\n[01:22] Темы и вопросы слушателей для 0120\n\n\n\nПоддержи DevZen Podcast на Patreon!\n\nЛог чата: https://gitter.im/DevZenRu/live/archives/2016/11/26\nГолоса выпуска: Валерий, Иван, Александр, Светлана и гость Ильдар\nФоновая музыка: Plastic3 — Corporate Rock Motivation Loop 4\n\n\n\n\nПодкаст: Скачать (64.2MB) \n\n \n'
    shownotes = get_shownotes_with_timestamps(shownotes_text)
    assert len(shownotes) == 11


def test_get_shownotes_should_return_shownotes_if_first_timestamp_is_not_zero():
    shownotes_text = '\n\nГоворим про возможность частично восстанавливать данные из базы; Королёве — фреймворке для SPA, работающих на сервере; поддержке языка Rust для всех IDE; конференции C++ Russia, ScalaUA, YAC; решаем, лучше ли Go, чем Rust для NTPSec; обсуждаем причины закрытия RethinkDB; так ли хороши смартфоны на малине; обсуждаем темы слушателей.\nШоу нотес:\n\n\n[00:01] Расспрашиваем гостя\n[00:03] pg_filedump: Partial data recovery (-D flag)\n\nhttps://git.postgresql.org/gitweb/?p=pg_filedump.git;a=commitdiff;h=52fa0201f97808d518c64bcb9696f2a350678aa5\nhttps://habrahabr.ru/company/postgrespro/blog/319770/\n\n\n[00:08] Korolev: Single-page applications running on the server side \n[00:40] Rust Language Server Alpha Release\n[00:43] Конференция C++ Russia, 24-25 февраля 2017, промо-код: DevZen-Podcast\n[00:44] Go лучше, чем Rust для NTPSec\n\nhttps://www.opennet.ru/opennews/art.shtml?num=45832\nhttps://blog.ntpsec.org/2017/01/18/rust-vs-go.html\nhttp://esr.ibiblio.org/?p=7294\nhttp://esr.ibiblio.org/?p=7303\n\n\n[00:59] Yet another Conference 2017\n[01:00] RethinkDB post-mortem\n[01:24] ZeroPhone gives Smartphones the Raspberry Pi\n\nhttps://hackaday.com/2017/01/18/zerophone-gives-smartphones-the-raspberry-pi/\nhttps://store.artlebedev.ru/fashion/hoody-witch-pocket-male/#112554\n\n\n[01:30] How Do I Declare a Function Pointer in C?\n[01:34] ScalaUA\n[01:35] В закладки — 24 Days of PureScript\n[01:38] Материалы для начинающих по локпику:\n\nhttp://www.e-reading.club/bookreader.php/134241/Rukovodstvo_MIT_po_otkryvaniyu_zamkov_otmychkoii.pdf\nhttps://www.youtube.com/user/bosnianbill\nhttp://www.banggood.com/ru/Transparent-Practice-Padlocks-with-12pcs-Unlocking-Lock-Pick-Set-Key-Extractor-Tool-Lock-Pick-Tools-p-1031199.html?rmmds=search\n\n\n[01:40] Темы и вопросы слушателей для 0126\n[01:53] Пара кулстори о прослушке на работе\n\n\n\nПоддержи DevZen Podcast на Patreon!\n\nЛог чата: https://gitter.im/DevZenRu/live/archives/2017/01/21\nГолоса выпуска: Светлана, Иван, Александр а также Алексей.\nФоновая музыка: Plastic3 — Corporate Rock Motivation Loop 4\n\n\n\n\nПодкаст: Скачать (65.2MB) \n\n \n'
    shownotes = get_shownotes_with_timestamps(shownotes_text)
    assert len(shownotes) == 15


def test_get_topics_should_return_topics():
    with open('./episode-0121.mp3-large.json', 'r', encoding='utf8') as f:
        data = json.load(f)
        segments = [Segment(**x) for x in data['segments']]
        transcription = Transcription(data['text'], segments, data['language'])
    shownotes_source = [
        (0, 'Благодарим наших замечательных патронов'),
        (120, 'Расспрашиваем гостей'),
        (4020, 'Вакансия от компании Netronix'),
        (4080, 'Эликсир 1.4 на подходе'),
        (4860, 'Positive Hack Days 23-24 мая'),
        (5160, 'Прошел Flink meetup Berlin'),
        (5520, 'Митап Elixir-разработчиков в Rambler&Co'),
        (5580, 'Global Game Jam 2017 Berlin Registration'),
        (5820, 'Strip Hack Moscow 2016'),
        (6420, 'Amazon Aurora Update – PostgreSQL Compatibility'),
        (6720, 'Убийца DO от Amazon'),
        (7080, 'Новые инстансы EC2 с программируемым железом'),
        (7560, 'Big latencies? It’s open season for kernel bug-hunting!'),
        (7860, 'The Last Of Us — Part II'),
        (7980, 'Темы и вопросы слушателей для 0121')
    ]
    shownotes = [Shownotes(*x) for x in shownotes_source]
    topics = get_topic_texts(transcription, shownotes)
    assert len(topics) == len(shownotes)


def test_get_topics():
    with open('./episode-0122.mp3-large.json', 'r', encoding='utf8') as f:
        data = json.load(f)
        segments = [Segment(**x) for x in data['segments']]
        transcription = Transcription(data['text'], segments, data['language'])

    shownotes_source = [
        (240, 'Расспрашиваем гостей'),
        (660, 'TensorFlow'),
        (2040, 'Книга DeepLearningBook'),
        (3960, 'How we structure our work and teams at Basecamp'),
        (4440, 'Очистите стол перед вашим интервью в Amazon'),
        (5220, 'Bloomberg снял видео про российское IT'),
        (5640, 'Writing an OS in Rust (blog series)'),
        (6240, 'В закладки — гид по htop для новичков. Python-3.6 что-то принес'),
        (6660, 'Платформа для AI от OpenAI'),
        (6780, 'Google опенсорсит тул для визуализации многомерных данных'),
        (6960, 'AI хакатон в Минске 11 декабря'),
        (7020, 'Темы и вопросы слушателей для 0122')
    ]
    shownotes = [Shownotes(*x) for x in shownotes_source]
    topics = get_topic_texts(transcription, shownotes)
    assert len(topics) == len(shownotes)


def test_get_topics_should_return_empty_if_last_timestamp_if_greater_than_last_segment_end():
    # 137 episode's last shownote has timestamp after the mp3's end
    with open('./episode-0137.mp3-large.json', 'r', encoding='utf8') as f:
        data = json.load(f)
        segments = [Segment(**x) for x in data['segments']]
        transcription = Transcription(data['text'], segments, data['language'])

    shownotes_source = [
        (13, 'Извиняемся за факап со звуком и перезалив'),
        (75, 'Про количество записей в RSS-ленте'),
        (89, 'Благодарим патронов'),
        (288, 'Baidu launches SwiftScribe, an app that transcribes audio with AI'),
        (551, 'Отчет о поездке в Нью-Йорк и конференции pgconf.us'),
        (1632, 'Kubernetes-1.6'),
        (2212, 'Отчет о 2600'),
        (2767, 'Как определить монтаж на фото: разоблачаем фейки, фотошоп и ретушь'),
        (3188, 'В закладки — Power Bank и Power Shield от Амперки + Arduino WiFi'),
        (3738, 'Digital Ocean: Introducing Monitoring'),
        (4244, 'Acolyer blog: Chronix: Long term storage and retrieval technology for anomaly detection in operational data'),
        (4746, 'Traceable commit for PostgreSQL 10'),
        (4945, 'Темы и вопросы слушателей'),
        (6003, 'В закладки — Wipe and reinstall a running Linux system via SSH, without rebooting'),
        (6063, 'В закладки — Adjustable DC Regulated Power Supply DIY Kit'),
        (6153, 'В закладки — DIY усилитель для наушников'),
        (6257, 'Расскажите про ваши проекты')
    ]
    shownotes = [Shownotes(*x) for x in shownotes_source]
    topics = get_topic_texts(transcription, shownotes)
    assert len(topics) == 0


def test_get_topics_for_single_timestamp():
    # episode 322 has single topic for entire episode
    with open('./episode-0137.mp3-large.json', 'r', encoding='utf8') as f:
        data = json.load(f)
        segments = [Segment(**x) for x in data['segments']]
        transcription = Transcription(data['text'], segments, data['language'])

    shownotes = [Shownotes(timestamp=151.0, title='Интервью с гостем — книга про юнит тестирование')]
    topics = get_topic_texts(transcription, shownotes)
    assert len(topics) == len(shownotes)


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

    sentences_with_timestamps = get_sentences_with_timestamps(transcription, lambda text: [x.text for x in pipeline(text).sentences])

    assert len(sentences_with_timestamps) == 4
    assert 0 == sentences_with_timestamps[0][0] and 5.42 == sentences_with_timestamps[0][1]
    assert 5.42 == sentences_with_timestamps[1][0] and 8.22 == sentences_with_timestamps[1][1]
    assert 8.22 == sentences_with_timestamps[2][0] and 10.22 == sentences_with_timestamps[2][1]
    assert 10.22 == sentences_with_timestamps[3][0] and 22.740000000000002 == sentences_with_timestamps[3][1]

    assert transcription is not None


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

    sentences_with_timestamps = get_sentences_with_timestamps(transcription, lambda text: [x.text for x in pipeline(text).sentences])

    assert len(sentences_with_timestamps) == 4
    assert 297.38 == sentences_with_timestamps[0][0] and 304.38 == sentences_with_timestamps[0][1]
    assert 304.38 == sentences_with_timestamps[1][0] and 307.38 == sentences_with_timestamps[1][1]
    assert 304.38 == sentences_with_timestamps[2][0] and 307.38 == sentences_with_timestamps[2][1]
    assert 307.38 == sentences_with_timestamps[3][0] and 314.38 == sentences_with_timestamps[3][1]



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

    sentences_with_timestamps = get_sentences_with_timestamps(transcription, lambda text: [x.text for x in pipeline(text).sentences])

    assert len(sentences_with_timestamps) == 5
    assert 297.38 == sentences_with_timestamps[0][0] and 304.38 == sentences_with_timestamps[0][1]
    assert 304.38 == sentences_with_timestamps[1][0] and 307.38 == sentences_with_timestamps[1][1]
    assert 304.38 == sentences_with_timestamps[2][0] and 307.38 == sentences_with_timestamps[2][1]
    assert 304.38 == sentences_with_timestamps[3][0] and 307.38 == sentences_with_timestamps[3][1]
    assert 307.38 == sentences_with_timestamps[4][0] and 314.38 == sentences_with_timestamps[4][1]


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

    sentences_with_timestamps = get_sentences_with_timestamps(
        transcription,
        lambda text: [x.text for x in pipeline(text).sentences])

    assert len(sentences_with_timestamps) == 3
    assert 307.38 == sentences_with_timestamps[0][0] and 314.38 == sentences_with_timestamps[0][1]
    assert 314.38 == sentences_with_timestamps[1][0] and 318.38 == sentences_with_timestamps[1][1]
    assert 314.38 == sentences_with_timestamps[2][0] and 325.38 == sentences_with_timestamps[2][1]


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

    sentences_with_timestamps = get_sentences_with_timestamps(
        transcription,
        lambda text: [x.text for x in pipeline(text).sentences])

    assert len(sentences_with_timestamps) == 4
    assert 167.38 == sentences_with_timestamps[0][0] and 170.38 == sentences_with_timestamps[0][1]
    assert 170.38 == sentences_with_timestamps[1][0] and 177.38 == sentences_with_timestamps[1][1]
    assert 177.38 == sentences_with_timestamps[2][0] and 178.38 == sentences_with_timestamps[2][1]
    assert 178.38 == sentences_with_timestamps[3][0] and 189.38 == sentences_with_timestamps[3][1]


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

    sentences_with_timestamps = get_sentences_with_timestamps(
        transcription,
        lambda text: [x.text for x in pipeline(text).sentences])

    assert len(sentences_with_timestamps) == 3
    assert 1632.38 == sentences_with_timestamps[0][0] and 1643.38 == sentences_with_timestamps[0][1]
    assert 1636.38 == sentences_with_timestamps[1][0] and 1643.38 == sentences_with_timestamps[1][1]
    assert 1644.38 == sentences_with_timestamps[2][0] and 1647.38 == sentences_with_timestamps[2][1]


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

    sentences_with_timestamps = get_sentences_with_timestamps(
        transcription,
        lambda text: [x.text for x in pipeline(text).sentences])

    assert len(sentences_with_timestamps) == 2
    assert 4092 == sentences_with_timestamps[0][0] and 4102 == sentences_with_timestamps[0][1]
    assert 4100 == sentences_with_timestamps[1][0] and 4102 == sentences_with_timestamps[1][1]


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

    sentences_with_timestamps = get_sentences_with_timestamps(
        transcription,
        lambda text: [x.text for x in pipeline(text).sentences])

    assert len(sentences_with_timestamps) == 4
    assert 4140 == sentences_with_timestamps[0][0] and 4144 == sentences_with_timestamps[0][1]
    assert 4142 == sentences_with_timestamps[1][0] and 4146 == sentences_with_timestamps[1][1]
    assert 4144 == sentences_with_timestamps[2][0] and 4150 == sentences_with_timestamps[2][1]
    assert 4148 == sentences_with_timestamps[3][0] and 4154 == sentences_with_timestamps[3][1]


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

    sentences_with_timestamps = get_sentences_with_timestamps(
        transcription,
        lambda text: [x.text for x in pipeline(text).sentences])

    assert len(sentences_with_timestamps) == 4
    assert 3431.92 == sentences_with_timestamps[0][0] and 3433.92 == sentences_with_timestamps[0][1] and 'Я понял.' == sentences_with_timestamps[0][2]
    assert 3431.92 == sentences_with_timestamps[1][0] and 3433.92 == sentences_with_timestamps[1][1] and 'То есть это не на 100%.' == sentences_with_timestamps[1][2]
    assert 3433.92 == sentences_with_timestamps[2][0] and 3437.92 == sentences_with_timestamps[2][1] and 'То есть это не 100% А, не 100%' == sentences_with_timestamps[2][2]
    assert 3437.92 == sentences_with_timestamps[3][0] and 3441.92 == sentences_with_timestamps[3][1] and 'С и не 100% отсутствие П. Вот это так можно расценить.' == sentences_with_timestamps[3][2]


def test_get_sentences_with_timestamps_111():
    # with open('/mnt/d/projects/podcast-shownotes/episodes/episode-0190.mp3-large.json', 'r', ) as f:
    #     data = json.load(f)
    #     segments = [Segment(**x) for x in data['segments']]
    #     transcription = Transcription(data['text'], segments, data['language'])

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

    assert len(sentences_with_timestamps) == 3


def test_unknown():
    # with open('/mnt/d/projects/podcast-shownotes/episodes/episode-0207.mp3-large.json', 'r', ) as f:
    #     data = json.load(f)
    #     segments = [Segment(**x) for x in data['segments']]
    #     transcription = Transcription(data['text'], segments, data['language'])

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
        lambda text: [x.text for x in pipeline(text).sentences],
        verbose=2)

    assert len(sentences_with_timestamps) == 3


def test_should_split_large_sentences_into_smaller_when_sentence_is_longer_than_50_words():
    segments_json = [
        { "id": 27, "seek": 20388, "start": 209.48, "end": 214.079, "text": " Да, конечно. Вот я сейчас скину в чат, прям такой, мне кажется максимально серьезное отношение к", "tokens": [], "temperature": 0, "avg_logprob": -0.288, "compression_ratio": 2.066, "no_speech_prob": 0.369 },
        { "id": 28, "seek": 20388, "start": 214.079, "end": 220.8, "text": " звукоизоляции, даже мне кажется серьезнее, чем у меня. Ну и давайте тогда может быть перейдем сразу к", "tokens": [], "temperature": 0, "avg_logprob": -0.288, "compression_ratio": 2.066, "no_speech_prob": 0.369 },
        { "id": 29, "seek": 20388, "start": 220.8, "end": 230.12, "text": " Сергею, а нет, нет, нет, у нас же первая тема, это что мы, чему мы научились за неделю. Ну можно и в таком", "tokens": [], "temperature": 0, "avg_logprob": -0.288, "compression_ratio": 2.066, "no_speech_prob": 0.369 },
        { "id": 30, "seek": 23012, "start": 230.12, "end": 241.480, "text": " порядке, конечно, я от себя добавил один маленький пунктик, я обратил внимание, что когда ты сидишь,", "tokens": [], "temperature": 0, "avg_logprob": -0.181, "compression_ratio": 1.974, "no_speech_prob": 0.487 },
        { "id": 31, "seek": 23012, "start": 241.480, "end": 251.6, "text": " ну у нас распределенная команда и основной канал коммуникации это слэк, и ну когда ты новый сотрудник", "tokens": [], "temperature": 0, "avg_logprob": -0.181, "compression_ratio": 1.974, "no_speech_prob": 0.487 },
        { "id": 32, "seek": 23012, "start": 251.6, "end": 257.32, "text": " в компании, то тебя особо никто ничего не пишет, потому что тебя все равно мало кто знает и так", "tokens": [], "temperature": 0, "avg_logprob": -0.181, "compression_ratio": 1.974, "no_speech_prob": 0.487 },
        { "id": 33, "seek": 25732, "start": 257.32, "end": 264.64, "text": " далее. Вот сейчас я в компании уже около полугода, чуть больше, и люди меня знают, и я обратил внимание,", "tokens": [], "temperature": 0, "avg_logprob": -0.279, "compression_ratio": 1.901, "no_speech_prob": 0.274 },
        { "id": 34, "seek": 25732, "start": 264.64, "end": 270.6, "text": " что мне стали достаточно часто писать в слэк, при том часто там какой-нибудь треки, где меня", "tokens": [], "temperature": 0, "avg_logprob": -0.279, "compression_ratio": 1.901, "no_speech_prob": 0.274 },
        { "id": 35, "seek": 25732, "start": 270.6, "end": 277.36, "text": " mention, о кстати Александр, а было бы вот клево, нам и такую штуку, загонтребьетесь куда-нибудь по", "tokens": [], "temperature": 0, "avg_logprob": -0.279, "compression_ratio": 1.901, "no_speech_prob": 0.274 },
        { "id": 36, "seek": 25732, "start": 277.36, "end": 283.76, "text": " пансорс, что ты об этом думаешь, такого рода вещи, и это начинает отвлекать меня, ну я первый раз", "tokens": [], "temperature": 0, "avg_logprob": -0.279, "compression_ratio": 1.901, "no_speech_prob": 0.274 },
        { "id": 37, "seek": 28376, "start": 283.76, "end": 292.039, "text": " работаю в полностью, ну как-то в remote first команде компании, и впервые столкнулся такой", "tokens": [], "temperature": 0, "avg_logprob": -0.227, "compression_ratio": 1.942, "no_speech_prob": 0.358 },
        { "id": 38, "seek": 28376, "start": 292.039, "end": 296.88, "text": " проблемой, что по сути ты оказываешься в одном огромном open space, где вот любой человек из", "tokens": [], "temperature": 0, "avg_logprob": -0.227, "compression_ratio": 1.942, "no_speech_prob": 0.358 },
        { "id": 39, "seek": 28376, "start": 296.88, "end": 300.56, "text": " компании, компания растет к тебе, вот в любой момент, любой может подойти и что-то написать.", "tokens": [], "temperature": 0, "avg_logprob": -0.227, "compression_ratio": 1.942, "no_speech_prob": 0.358 }
    ]

    transcription = Transcription(
        text=' Да, конечно. Вот я сейчас скину в чат, прям такой, мне кажется максимально серьезное отношение к звукоизоляции, даже мне кажется серьезнее, чем у меня. Ну и давайте тогда может быть перейдем сразу к Сергею, а нет, нет, нет, у нас же первая тема, это что мы, чему мы научились за неделю. Ну можно и в таком порядке, конечно, я от себя добавил один маленький пунктик, я обратил внимание, что когда ты сидишь, ну у нас распределенная команда и основной канал коммуникации это слэк, и ну когда ты новый сотрудник в компании, то тебя особо никто ничего не пишет, потому что тебя все равно мало кто знает и так далее. Вот сейчас я в компании уже около полугода, чуть больше, и люди меня знают, и я обратил внимание, что мне стали достаточно часто писать в слэк, при том часто там какой-нибудь треки, где меня mention, о кстати Александр, а было бы вот клево, нам и такую штуку, загонтребьетесь куда-нибудь по пансорс, что ты об этом думаешь, такого рода вещи, и это начинает отвлекать меня, ну я первый раз работаю в полностью, ну как-то в remote first команде компании, и впервые столкнулся такой проблемой, что по сути ты оказываешься в одном огромном open space, где вот любой человек из компании, компания растет к тебе, вот в любой момент, любой может подойти и что-то написать.',
        segments=[Segment(**x) for x in segments_json],
        language='ru')

    sentences_with_timestamps = get_sentences_with_timestamps_by_letter(
        transcription,
        lambda text: [x.text for x in pipeline(text).sentences],
        verbose=2)

    assert len(sentences_with_timestamps) == 8


# @pytest.mark.skip('explicit text')
def test_get_sentences_for_358():
    with open('/mnt/d/projects/podcast-shownotes/episodes/episode-0358.mp3-medium.json', 'r', ) as f:
        data = json.load(f)
        segments = [Segment(**x) for x in data['segments']]
        transcription = Transcription(data['text'], segments, data['language'])

    sentences_with_timestamps = get_sentences_with_timestamps_by_letter(
        transcription,
        lambda text: [x.text for x in pipeline(text).sentences],
        verbose=2)

    assert len(sentences_with_timestamps) == 8
