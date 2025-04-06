# Why not use langchain or other frameworks?

While I don't want to rule out the use of some middle framework down the line, I don't want to be limited by what is provided by the framework when there are libraries that exist to achieve the same thing outside the LLM context as well. Further, Langchain does not have very user friendly documentation and seems needlessly complex. I don't want to bring in the entire stack if I just want to
use a very small piece of it.

Case where using a framework is useful? In the current scenario, while I am focusing on using Ollama
with the code (and hence using ollama-python bindings), it is possible that I may want to fiddle around with other LLM providers as well. Under such circumstances, a library that provides a generic
interface to all providers would be good, but only that. Even there, I hope to have a class based abstraction which would allow me to slipstream different components without affecting the remainder of the codebase.

# Why use LLMs at all?

LLMs are in vogue, but we have techniques that have been developed and matured before LLMs. So, why bring a sword to a gunfight? When using Ollama with Llama3.2, there were articles that it just did not detect as opinion pieces. So, why do I even want to bank on a whimsical LLM when a defined algorithm can do that? Besides, using an LLM there is essentially using a black box without understanding the principles behind it. Sure, LLM may find a use somewhere.

