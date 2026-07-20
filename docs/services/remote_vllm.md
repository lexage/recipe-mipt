# Запуск vLLM на удалённом кластере
Чтобы настроить у себя доступ, нужно:

## 1. Настроить SSH-доступ к кластеру

### Шаг 1.1. Добавить хост кластера в ssh-config

Открыть (или создать) файл `~/.ssh/config` и добавить туда:

```
Host mipt_aigrant_cluster
  HostName proxy2.cod.phystech.edu
  Port 10209
  User <username>
  IdentityFile ~/.ssh/mipt/opt_mipt
  IdentitiesOnly yes
  ServerAliveInterval 60
  ServerAliveCountMax 3
  TCPKeepAlive yes
  Compression yes
```

`<username>` — ваш логин на кластере.

С таким конфигом подключение уже работает, но при каждом `ssh` придётся вводить пароль пользователя. Чтобы этого избежать, нужно сгенерировать пару SSH-ключей (шаги 1.2–1.4).

### Шаг 1.2. Сгенерировать пару SSH-ключей

Пара состоит из приватного ключа (хранится только на вашем компьютере, никому не передаётся) и публичного (кладётся на сервер). Рекомендуемый алгоритм — `ed25519`:

```shell
ssh-keygen -t ed25519 -f ~/.ssh/opt_mipt
```

Команда создаст два файла:

- `~/.ssh/opt_mipt` — приватный ключ;
- `~/.ssh/opt_mipt.pub` — публичный ключ.

> **NOTE:** При генерации будет предложено задать пароль (passphrase) на приватный ключ. Это необязательно, но рекомендуется для дополнительной защиты.

### Шаг 1.3. Скопировать публичный ключ на сервер

Проще всего через `ssh-copy-id` (потребуется один раз ввести пароль пользователя):

```shell
ssh-copy-id -i ~/.ssh/opt_mipt -p 10209 <username>@proxy2.cod.phystech.edu
```

Порт зависит от выданного вам узла — это может быть `10209`, `10096` или `10210`.

Команда сама добавит содержимое публичного ключа в `~/.ssh/authorized_keys` на сервере.

**Альтернатива, если `ssh-copy-id` недоступен** (например, на Windows) — скопировать ключ вручную:

```shell
# 1. Показать содержимое публичного ключа и скопировать его
cat ~/.ssh/opt_mipt.pub

# 2. Подключиться к серверу по паролю
ssh -p 10209 <username>@proxy2.cod.phystech.edu

# 3. Уже на сервере — добавить ключ и выставить права
mkdir -p ~/.ssh
echo "ваш_публичный_ключ" >> ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys
chmod 700 ~/.ssh
```

### Шаг 1.4. Положить ключ туда, где его ждёт конфиг

В конфиге из шага 1.1 указан путь `~/.ssh/mipt/opt_mipt`, поэтому переносим ключи в папку `~/.ssh/mipt`:

```shell
mkdir -p ~/.ssh/mipt
mv ~/.ssh/opt_mipt ~/.ssh/opt_mipt.pub ~/.ssh/mipt/
chmod 600 ~/.ssh/mipt/opt_mipt
```

После этого подключение проходит без ввода пароля:

```shell
ssh mipt_aigrant_cluster
```

## 2. Подключиться к контейнеру на кластере 

```shell
ssh mipt_aigrant_recipe
```

## 3. Запустить модель

> **NOTE:** 
> Все действия в этом пункте стоит совершать в отдельной треминальной сессии. Можно использовать утилиту `screen`. 
>
> >Создать новую сессию
> >```shell
> >screen -S session_name
> >```
> 
>> Просмотр активных сессий
>> ```shell
>> screen -ls
>> ```
>
>> Подключение к активной сессии
>> ```shell
>> screen -r session_name
>> ```
>
>> Отсоединение от сессии
>> ```shell
>> Ctrl-a d
>> ```
>


### 3.1. Скрипт для запуска vllm лежит в /work/vllm_exp
```shell
cd /data/work/team_recipe/work/vllm-exp
```

### 3.2. Активируем окружение
```shell
. .venv/bin/activate
```
> **NOTE:** Иногда после активации окружения нужно будет делать `uv sync`

### 3.3. Запуск vllm, используя скрипт
```shell
CUDA_VISIBLE_DEVICES=0,1 vllm serve Qwen/Qwen1.5-32B-Chat-AWQ --port 7215 --max-num-batched-tokens 8192 --tensor-parallel-size 2
```

## 4. Пробросить порты на локальную машину

У себя на локалке: 
```shell
ssh -N -L localhost:7215:0.0.0.0:7215 -L localhost:7216:0.0.0.0:7216 mipt_aigrant_recipe
```
> **NOTE:** `localhost:<local_port>:0.0.0.0:<model_port>`
> - _local_port_ : порт по которому модель будет доступна на локалке, может быть любым;
> - _model_port_ : порт, на котором развернута модель на удалённом кластере; 

## 5. Profit

> **NOTE:** В этом пунтке `7215` - это _local_port_

Можно обращаться к модели через API, напр. используя OpenAI клинет на Python:

```python
from openai import OpenAI

MODEL_NAME = "your_model"
PROMPT = "prompt"
URL = "http://localhost:7215/v1"

client = OpenAI(
    base_url="",
    api_key="vllm"
)

response = self.client.chat.completions.create(
    model=self.model_name,
    messages=[
        {"role": "user", "content": PROMPT}],
    temperature=0.7,
)
```

Можно открыть сваггер по ссылке:
```
http://localhost:7215/docs
```
