package main

import (
	"net/http"

	"example.test/verdant-port/internal/relay"
)

func main() {
	http.ListenAndServe(":8080", relay.NewRouter())
}
