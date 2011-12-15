# 
# Do NOT Edit the Auto-generated Part!
# Generated by: spectacle version 0.23
# 
# >> macros
%define __strip /bin/true
%define _build_name_fmt    %%{ARCH}/%%{NAME}-%%{VERSION}-%%{RELEASE}.%%{ARCH}.dontuse.rpm
%global debug_package %{nil}
# << macros

Name:       qemu-usermode-static
Summary:    Universal CPU emulator
Version:    1.0.2011.12
Release:    1
Group:      System/Emulators/PC
License:    GPLv2
URL:        http://qemu.org
Source0:    qemu-linaro-1.0-2011.12.tar.gz
Source1:    qemu-binfmt-conf.sh
Source100:  qemu-usermode-static.yaml
BuildRequires:  pkgconfig(ext2fs)
BuildRequires:  pkgconfig(glib-2.0)
BuildRequires:  bison
BuildRequires:  curl-devel
BuildRequires:  zlib-static
BuildRequires:  glibc-static
BuildRequires:  python-devel
BuildRequires:  glib2-static
BuildRequires:  zlib-devel
ExclusiveArch: %{ix86}
AutoReqProv:   0


%description
QEMU is an extremely well-performing CPU emulator that allows you to choose between simulating an entire system and running userspace binaries for different architectures under your native operating system. It currently emulates x86, ARM, PowerPC and SPARC CPUs as well as PC and PowerMac systems.



%prep
%setup -q -n qemu-linaro-1.0-2011.12

# >> setup
# << setup

%build
# >> build pre

CFLAGS=`echo $CFLAGS | sed 's|-fno-omit-frame-pointer||g'` ; export CFLAGS ;
CFLAGS=`echo $CFLAGS | sed 's|-O2|-O|g'` ; export CFLAGS ;


./configure \
--prefix=/usr \
--sysconfdir=%_sysconfdir \
--static \
--interp-prefix=/usr/share/qemu/qemu-i386 \
--disable-system \
--enable-linux-user \
--enable-guest-base \
--disable-werror \
--target-list=arm-linux-user,mipsel-linux-user
# << build pre


make %{?jobs:-j%jobs}

# >> build post
# << build post
%install
rm -rf $RPM_BUILD_ROOT
# >> install pre
mkdir -p %{buildroot}/usr/sbin
install -m 755 %{SOURCE1} $RPM_BUILD_ROOT/usr/sbin
# << install pre
%if 0%{?moblin_version}
%make_install
%else
%makeinstall
%endif

# >> install post
#slint checking rpm package causes error if these files are packaged
# remove now as they are not needed for OBS/CE servers.
rm -f $RPM_BUILD_ROOT/usr/share/qemu/openbios-ppc
rm -f $RPM_BUILD_ROOT/usr/share/qemu/openbios-sparc32
rm -f $RPM_BUILD_ROOT/usr/share/qemu/openbios-sparc64
rm -rf $RPM_BUILD_ROOT/etc
shellquote()
{
    for arg; do
        arg=${arg//\\/\\\\}
#        arg=${arg//\$/\$}   # already needs quoting ;(
#        arg=${arg/\"/\\\"}  # dito
#        arg=${arg//\`/\`}   # dito
        arg=${arg//\\|/\|}
        arg=${arg//\\|/|}
        echo "$arg"
    done
}

echo "Creating baselibs_new.conf"
echo ""
rm -rRf /tmp/baselibs_new.conf || true
shellquote "%{name}" >> /tmp/baselibs_new.conf
shellquote "  targettype x86 block!" >> /tmp/baselibs_new.conf
shellquote "  targettype 32bit block!" >> /tmp/baselibs_new.conf
shellquote "  targettype arm autoreqprov off" >> /tmp/baselibs_new.conf
shellquote "  targettype arm extension -arm" >> /tmp/baselibs_new.conf
shellquote "  targettype arm +/" >> /tmp/baselibs_new.conf
shellquote "  targettype arm -%{_mandir}" >> /tmp/baselibs_new.conf
shellquote "  targettype arm -%{_docdir}" >> /tmp/baselibs_new.conf

cat /tmp/baselibs_new.conf >> %{_sourcedir}/baselibs.conf
cat %{_sourcedir}/baselibs.conf

rm -rf $RPM_BUILD_ROOT/%{_datadir}
ls -lR $RPM_BUILD_ROOT

# << install post

%files
%defattr(-,root,root,-)
%{_bindir}/qemu-arm
%{_bindir}/qemu-mipsel
%{_sbindir}/qemu-binfmt-conf.sh
# >> files
# << files


