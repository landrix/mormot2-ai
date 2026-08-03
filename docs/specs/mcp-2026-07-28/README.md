# MCP-Spezifikation 2026-07-28 (lokale Kopie)

**Unveränderte Kopie** der Protokoll-Spezifikation, gegen die `mormot.ai.mcp.*` gebaut ist.
Sie liegt hier, damit Review- und Audit-Läufe gegen den **Normtext** belegen können statt
gegen Erinnerung — die häufigste Fehlerquelle bei Protokollarbeit ist nicht der Code, es
ist der Prüfer, der eine Anforderung erfindet oder eine ältere Revision im Kopf hat.

Genau darum ist die Revision hier heikel: **2026-07-28 ist stateless**. `initialize`,
Protokoll-Sessions, Batching, SSE-Resumability und der HTTP+SSE-Transport sind
**gelöscht**; jede Anfrage trägt ihre Protokollversion und die Client-Capabilities in
`params._meta`. Wer aus älterer MCP-Kenntnis prüft, meldet systematisch Falsches.

## Herkunft

| | |
|---|---|
| Quelle | https://github.com/modelcontextprotocol/modelcontextprotocol |
| Pfad | `docs/specification/2026-07-28/` |
| Commit | `db4bfcff3d60f5df01a21bdf6b78f7012cac4634` (2026-07-28) |
| Geholt am | 2026-08-03 |
| Umfang | 31 `.mdx`-Dateien inkl. `schema.mdx` (die vollständige TypeScript-Schema-Referenz) |

## Lizenz

Siehe [LICENSE](LICENSE) — wortgleich aus dem Quell-Repo übernommen. Kurzfassung: Das
MCP-Projekt ist mitten in einem Lizenzwechsel von MIT auf **Apache-2.0**; neue
Spezifikations-Beiträge stehen unter Apache-2.0, Altbeiträge ohne Relizenzierungs-Zustimmung
bleiben MIT. Beide erlauben die unveränderte Weitergabe mit Lizenzhinweis.

**Diese Dateien sind nicht Teil von LandrixAI** und stehen **nicht** unter dessen
MPL/GPL/LGPL-Dreifachlizenz. Sie sind eine unveränderte Beilage, kein abgeleitetes Werk —
deshalb die eigene `LICENSE` in diesem Verzeichnis und der Eintrag im
[NOTICE](../../../NOTICE) der Extension. **Nicht editieren**: eine veränderte Spec wäre als
Prüfmaßstab wertlos und lizenzrechtlich eine Ableitung.

## Aktualisieren

Erscheint eine neue Revision, wird sie **daneben** abgelegt (`mcp-<datum>/`), nicht
darüber — die alte bleibt der Maßstab, gegen den der aktuelle Code gebaut wurde, bis er
umgestellt ist. Nach dem Umstellen die alte Revision entfernen und
[DESIGN.md](../../../DESIGN.md) („Protokoll") mitziehen.

```bash
BASE=https://raw.githubusercontent.com/modelcontextprotocol/modelcontextprotocol/main/docs/specification/<revision>
# je Datei: curl -sfS -o <ziel> "$BASE/<pfad>"
# Commit-Pin für die Tabelle oben:
curl -s "https://api.github.com/repos/modelcontextprotocol/modelcontextprotocol/commits?path=docs/specification/<revision>&per_page=1"
```

## Einstieg beim Prüfen

| Frage | Datei |
|---|---|
| Was hat sich gegenüber der Vorrevision geändert? | [changelog.mdx](changelog.mdx) |
| Was ist *deprecated* und darf **nicht** neu gebaut werden? | [deprecated.mdx](deprecated.mdx) |
| Wie sieht eine Nachricht genau aus (Felder, Typen)? | [schema.mdx](schema.mdx) |
| Auth-Pflichten des Servers | [basic/authorization/](basic/authorization/) |
| Transport (Streamable HTTP, stdio) | [basic/transports/](basic/transports/) |
| Pattern: MRTR, Progress, Cancellation, Subscriptions | [basic/patterns/](basic/patterns/) |
| Werkzeuge/Ressourcen/Prompts | [server/](server/) |
| Pagination, Completion, Caching, Logging | [server/utilities/](server/utilities/) |

Die OAuth-Normtexte, auf die die Auth-Kapitel verweisen (RFC 9728, 8414, 7591, 6749 …),
liegen auf der Landrix-Seite: `docs/specs/oauth/` im Hauptrepo.
