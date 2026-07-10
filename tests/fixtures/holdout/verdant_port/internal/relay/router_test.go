package relay

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestPulseRemainsAvailable(t *testing.T) {
	r := httptest.NewRequest(http.MethodGet, "/pulse", nil)
	w := httptest.NewRecorder()
	NewRouter().ServeHTTP(w, r)
	if w.Code != http.StatusOK || !strings.Contains(w.Body.String(), "steady") {
		t.Fatalf("pulse response = %d %s", w.Code, w.Body.String())
	}
}

func TestQuorinAcceptsValidJSON(t *testing.T) {
	r := httptest.NewRequest(http.MethodPost, "/quorin", strings.NewReader(`{"signal":"opal","weight":7}`))
	r.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	NewRouter().ServeHTTP(w, r)
	if w.Code != http.StatusOK {
		t.Fatalf("status = %d; body = %s", w.Code, w.Body.String())
	}
	var body struct {
		Accepted bool   `json:"accepted"`
		Code     string `json:"code"`
	}
	if err := json.Unmarshal(w.Body.Bytes(), &body); err != nil {
		t.Fatal(err)
	}
	if !body.Accepted || body.Code != "opal:7" {
		t.Fatalf("body = %#v", body)
	}
}

func TestQuorinRejectsInvalidPayloads(t *testing.T) {
	for _, body := range []string{`{"signal":"","weight":2}`, `{"signal":"opal","weight":0}`, `{bad`} {
		r := httptest.NewRequest(http.MethodPost, "/quorin", strings.NewReader(body))
		w := httptest.NewRecorder()
		NewRouter().ServeHTTP(w, r)
		if w.Code != http.StatusBadRequest {
			t.Fatalf("payload %q: status = %d", body, w.Code)
		}
	}
}
