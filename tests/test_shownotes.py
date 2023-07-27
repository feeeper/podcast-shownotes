import json

from src.shownotes import get_shownotes_with_timestamps, get_topic_texts
from src.transcription import Transcription, Segment, Shownotes


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
    print(f'{len(topics)=}\t{len(shownotes)=}')
    for i in range(len(shownotes)):
        print(f'{topics[i][0]}\t{shownotes[i].timestamp}')
    assert len(topics) == len(shownotes)


def test_get_topics_if_first_topic_start_not_from_the_beginning():
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
    print(f'{len(topics)=}\t{len(shownotes)=}')
    for i in range(len(shownotes)):
        print(f'{topics[i][0]}\t{shownotes[i].timestamp}')
    assert len(topics) == len(shownotes)
