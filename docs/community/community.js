const inviteImage = document.querySelector("[data-community-invite-image]");
const invitePlaceholder = document.querySelector("[data-community-placeholder]");
const inviteUpdated = document.querySelector("[data-community-updated]");
let contactEnabled = true;

const showContactImage = (visible) => {
  inviteImage.hidden = !visible;
  invitePlaceholder.hidden = visible;
};

inviteImage.addEventListener("load", () => showContactImage(contactEnabled));
inviteImage.addEventListener("error", () => showContactImage(false));
if (inviteImage.complete && !inviteImage.naturalWidth) showContactImage(false);

const formatDate = (value) => {
  if (!value) return "Not available / 暂无信息";
  const date = new Date(`${value}T00:00:00Z`);
  if (Number.isNaN(date.getTime())) return "Not available / 暂无信息";
  const english = new Intl.DateTimeFormat("en", {
    year: "numeric",
    month: "short",
    day: "numeric",
    timeZone: "UTC",
  }).format(date);
  const chinese = new Intl.DateTimeFormat("zh-CN", {
    year: "numeric",
    month: "long",
    day: "numeric",
    timeZone: "UTC",
  }).format(date);
  return `${english} / ${chinese}`;
};

fetch("../assets/community-invite.json", { cache: "no-store" })
  .then((response) => {
    if (!response.ok) throw new Error("Contact metadata unavailable");
    return response.json();
  })
  .then((contact) => {
    inviteUpdated.textContent = formatDate(contact.updated);
    contactEnabled = contact.status === "active" && contact.contactType === "personal-wechat" && Boolean(contact.image);
    if (!contactEnabled) {
      showContactImage(false);
      return;
    }

    inviteImage.src = `${contact.image}?updated=${encodeURIComponent(contact.updated)}`;
  })
  .catch(() => {
    inviteUpdated.textContent = "Metadata unavailable / 更新信息暂不可用";
  });
