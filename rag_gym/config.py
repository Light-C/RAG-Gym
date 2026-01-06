config_openai = {
	"api_type": "openai",
	"api_base": None,
	"api_version": None,
	"api_key": None
}

config_azure = {
	"base_url": None,
	"api_key": None
}

config_local_llm = {
	"base_url": "http://localhost:8333/v1",
	"api_key": "empty",
	"model_name": "/Qwen3-Next-80B-A3B-Instruct-int4-mixed-AutoRound"
}

config_local_embedding = {
	"base_url": "http://localhost:8001/v1",
	"api_key": "empty",
	"model_name": "bge-m3"
}
