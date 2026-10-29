/**
 * Magic: The Gathering tools the app runs itself, straight from the browser:
 * card text from Scryfall's API, and combos and Commander deck stats from
 * EDHREC's page data (EDHREC's combos come from Commander Spellbook). EDHREC
 * has no official API, so its page data could change without notice. Each
 * result starts with numbered `[n] Title — URL` lines, shown as sources.
 */
import { toolDefinition } from "@tanstack/ai"
import { useMemo } from "react"

import { mtgToolsEnabledStore } from "#/lib/settings"
import { shorten } from "#/lib/sources"

type ScryfallFace = {
  name: string
  mana_cost?: string
  type_line?: string
  oracle_text?: string
  power?: string
  toughness?: string
  loyalty?: string
}
type ScryfallCard = ScryfallFace & {
  scryfall_uri: string
  card_faces?: Array<ScryfallFace>
  legalities?: Record<string, string>
}

/** The card Scryfall finds for a name; misspellings are fine. */
async function findCard(name: string) {
  const res = await fetch(
    `https://api.scryfall.com/cards/named?fuzzy=${encodeURIComponent(name)}`
  )
  const card = (await res.json()) as ScryfallCard & { details?: string }
  if (!res.ok) throw new Error(card.details ?? `No card named "${name}".`)
  return card
}

const pageUrl = (uri: string) => uri.split("?")[0]

/** A card's front-face name the way EDHREC writes it in its URLs. */
const edhrecSlug = (name: string) =>
  name
    .split(" // ")[0]
    .normalize("NFD")
    .replace(/[̀-ͯ]/g, "")
    .toLowerCase()
    .replace(/[^a-z0-9\s-]/g, "")
    .trim()
    .replace(/\s+/g, "-")

type EdhrecCardView = {
  name: string
  num_decks?: number
  potential_decks?: number
}
type EdhrecList = {
  header: string
  href?: string
  cardviews: Array<EdhrecCardView>
  combo?: { count?: number; results?: Array<string> }
}

async function edhrecPage(path: string) {
  const res = await fetch(`https://json.edhrec.com/pages/${path}.json`)
  if (!res.ok) return null
  const page = (await res.json()) as {
    container?: {
      json_dict?: {
        cardlists?: Array<EdhrecList>
        card?: { num_decks?: number }
      }
    }
  }
  return page.container?.json_dict ?? null
}

function faceText(face: ScryfallFace) {
  const stats = face.loyalty
    ? `Loyalty ${face.loyalty}`
    : face.power
      ? `${face.power}/${face.toughness}`
      : ""
  return [
    [face.name, face.mana_cost].filter(Boolean).join(" "),
    [face.type_line, stats].filter(Boolean).join(" · "),
    face.oracle_text
  ]
    .filter(Boolean)
    .join("\n")
}

async function cardInfo(input: unknown) {
  const card = await findCard((input as { name: string }).name)
  const faces = card.card_faces?.length ? card.card_faces : [card]
  const commander = card.legalities?.commander
  return [
    `[1] ${card.name} on Scryfall — ${pageUrl(card.scryfall_uri)}`,
    ...faces.map(faceText),
    commander && `Commander: ${commander.replace("_", " ")}`
  ]
    .filter(Boolean)
    .join("\n\n")
}

/** Combos kept from a card's EDHREC combo page, most played first. */
const MAX_COMBOS = 8

async function findCombos(input: unknown) {
  const card = await findCard((input as { card: string }).card)
  const slug = edhrecSlug(card.name)
  const page = await edhrecPage(`combos/${slug}`)
  const combos = (page?.cardlists ?? []).filter((list) => list.combo)
  if (combos.length === 0) return `No combos found for ${card.name}.`
  return combos
    .slice(0, MAX_COMBOS)
    .map((list, i) => {
      const decks = list.combo?.count
      return [
        `[${i + 1}] ${list.header} — https://edhrec.com${list.href ?? `/combos/${slug}`}`,
        (list.combo?.results ?? []).join(", "),
        decks ? `In ${decks.toLocaleString("en-US")} decks.` : ""
      ]
        .filter(Boolean)
        .join("\n")
    })
    .join("\n\n")
}

/** EDHREC lists kept for a commander, and how many cards from each. */
const COMMANDER_LISTS: Record<string, number> = {
  "Top Cards": 10,
  "High Lift Cards": 8,
  Creatures: 8,
  Instants: 6,
  Sorceries: 6,
  Enchantments: 5,
  Planeswalkers: 3,
  "Utility Artifacts": 5,
  "Mana Artifacts": 4,
  "Utility Lands": 5
}

async function commanderCards(input: unknown) {
  const card = await findCard((input as { commander: string }).commander)
  const slug = edhrecSlug(card.name)
  const page = await edhrecPage(`commanders/${slug}`)
  if (!page?.cardlists?.length) {
    return `EDHREC has no Commander decks led by ${card.name}.`
  }
  const lists = page.cardlists
    .filter((list) => list.header in COMMANDER_LISTS)
    .map((list) => {
      const cards = list.cardviews
        .slice(0, COMMANDER_LISTS[list.header])
        .map((c) =>
          c.num_decks && c.potential_decks
            ? `${c.name} (${Math.round((100 * c.num_decks) / c.potential_decks)}%)`
            : c.name
        )
      const header =
        list.header === "High Lift Cards" ? "High synergy" : list.header
      return `${header}: ${cards.join(", ")}`
    })
  const decks = page.card?.num_decks
  return shorten(
    [
      `[1] ${card.name} on EDHREC — https://edhrec.com/commanders/${slug}`,
      decks
        ? `${decks.toLocaleString("en-US")} decks. Percentages are how many of them play each card.`
        : "",
      ...lists
    ]
      .filter(Boolean)
      .join("\n")
  )
}

/** One required text argument, and nothing else. */
const oneArg = (name: string, description: string) => ({
  type: "object" as const,
  properties: { [name]: { type: "string" as const, description } },
  required: [name],
  additionalProperties: false
})

const mtgTools = [
  toolDefinition({
    name: "card_info",
    description: "Get a Magic: The Gathering card's rules text.",
    inputSchema: oneArg("name", "The card's name.")
  }).client(cardInfo),
  toolDefinition({
    name: "find_combos",
    description: "Find Magic: The Gathering combos that use a card.",
    inputSchema: oneArg("card", "The card's name.")
  }).client(findCombos),
  toolDefinition({
    name: "commander_cards",
    description: "Get the most played cards in a commander's decks.",
    inputSchema: oneArg("commander", "The commander's name.")
  }).client(commanderCards)
]

/** The MTG tools when they're on in Settings, else none. */
export function useMtgTools() {
  const enabled = mtgToolsEnabledStore.use()
  return useMemo(() => (enabled ? mtgTools : []), [enabled])
}
