# Запуск vLLM на удалённом кластере
Чтобы настроить у себя доступ, нужно:

## 1. Добавить SSH конфиги

### MacOS/Linux

#### 1.1. Добавить в свой ssh-config (~/.ssh/config) вот это

```
Host mipt_aigrant_cluster
  HostName proxy2.cod.phystech.edu
  Port 10209
  User mipt-user
  IdentityFile ~/.ssh/mipt/rsa_asap
  IdentitiesOnly yes

Host mipt_aigrant_asap
  HostName 127.0.0.1
  Port 2222
  User dev
  ProxyJump mipt_aigrant_cluster
  IdentityFile ~/.ssh/mipt/rsa_asap
  IdentitiesOnly yes
  ServerAliveInterval 30
  ServerAliveCountMax 4
  HostKeyAlias mipt_aigrant_asap
  LocalForward localhost:3000 172.18.0.2:3000
  LocalForward localhost:3001 172.18.0.2:3001

Host mipt_aigrant_recipe
  HostName 127.0.0.1
  Port 2223
  User dev
  ProxyJump mipt_aigrant_cluster
  IdentityFile ~/.ssh/mipt/rsa_asap
  IdentitiesOnly yes
  ServerAliveInterval 30
  ServerAliveCountMax 4
  HostKeyAlias mipt_aigrant_asap
  LocalForward localhost:3002 172.18.0.3:3002
  LocalForward localhost:3003 172.18.0.3:3003
```

#### 1.2. Добавить файл с ключом в папку `~/.ssh/mipt/rsa_asap`. 

Ключ можно найти в общем тг канале исследования. 
Либо попросить у tg: @german_deer

### Windows
...


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
cd /work/vllm-exp/
```

### 3.2. Активируем окружение
```shell
. .venv/bin/activate
```
> **NOTE:** Иногда после активации окружения нужно будет делать `uv sync`

### 3.3. Запуск vllm, используя скрипт
```shell
CUDA_VISIBLE_DEVICES=0,1 ./vllm_run.sh Qwen/Qwen1.5-32B-Chat-AWQ --port 7215 --max-num-batched-tokens 8192 --tensor-parallel-size 2
```

## 4. Пробросить порты на локальную машину

У себя на локалке: 
```shell
ssh -N -L localhost:7215:172.18.0.3:7215 -L localhost:7216:172.18.0.3:7216 mipt_aigrant_recipe
```
> **NOTE:** `localhost:<local_port>:172.18.0.3:<model_port>`
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
